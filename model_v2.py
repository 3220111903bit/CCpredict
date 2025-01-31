import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class MedalPredictionModel(nn.Module):
    def __init__(self, m_years):
        super(MedalPredictionModel, self).__init__()

        self.m_years = m_years

        # Define the learnable parameters
        self.linear_C = nn.Linear(15, 6)
        self.linear_A = nn.Linear(15, 6)
        self.host_effect = nn.Linear(12, 12)  # Host effect

        # Define the self-attention layer (in this case, for attention between countries)
        self.attn = nn.MultiheadAttention(embed_dim=12, num_heads=1, batch_first=True)  # Attention between countries
        
        # Define the final transformation matrix (for output (c, 1, 4))
        self.ffn = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 12)
        )
        
        self.norm1 = nn.LayerNorm(12)
        self.norm2 = nn.LayerNorm(12)

        self.linear_final = nn.Linear(12, 3)

        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, country_data, athlete_data, athlete_to_country, host_country_index):
        a, c, m = athlete_data.shape[0], country_data.shape[0], country_data.shape[1]
        # country_data: Tensor with shape (c, m, 4) - where "c" is total countries
        # athlete_data: Tensor with shape (a, m, 4)
        # athlete_to_country: Shape (c, a), mapping athletes to countries (only contain 0 and 1)
        # host_country_index: Shape (c, 1), Index of the host country

        country_data = country_data.reshape(c, -1)
        country_data = self.linear_C(country_data) # shape (c, 8)

        athlete_data = athlete_data.reshape(a, -1)
        athlete_data = self.linear_A(athlete_data) # shape (a, 8)

        country_athlete_contribution = torch.matmul(athlete_to_country.float(), athlete_data) # shape (c, 8)

        # Combine country data with the athlete contributions
        combined_data = torch.cat((country_data, country_athlete_contribution), dim=1) # shape (c, 16)

        # Add the host effect
        if host_country_index:
            host_country_data = self.host_effect(combined_data[host_country_index])
    
            # Recombine the transformed host country data with the rest of the combined data
            combined_data = combined_data.clone()  # Make a copy to avoid in-place modification
            combined_data[host_country_index] = host_country_data
        
        activated_data = F.leaky_relu(combined_data, negative_slope=0.1)

        # Step 1: Apply self-attention on the (c, m_years) matrix for the current medal type
        attn_output, _ = self.attn(activated_data, activated_data, activated_data)  # Shape: (c, m_years*4)

        # Now, `attn_output` has shape (c, m_years, 4), which incorporates country-country interactions for each medal type
        attn_output = attn_output + activated_data

        ffn_output = self.ffn(attn_output)

        total_output = ffn_output + attn_output

        final_output = self.linear_final(self.dropout(total_output))

        return final_output

m_years = 5

import json
import numpy as np

with open('transformed_olympic_medals.json', 'r') as file:
    country_data = json.load(file)
# year -> country -> medal

with open('transformed_athlete_medals.json', 'r') as file:
    athlete_data = json.load(file)
# year -> country -> athlete -> medal

with open('host_history.json', 'r') as file:
    host_history = json.load(file)

assert set(country_data.keys()) == set(athlete_data.keys())

data_list = []
test_dict = {}

total_years = sorted(country_data.keys())

for i, year in enumerate(total_years + ['2028']):
    if i < 5:
        continue

    if year != '2028':
        all_countries = sorted(country_data[year].keys())
    elif '2024' in total_years:
        all_countries = sorted(country_data['2024'].keys())

    country_record = np.zeros((len(all_countries), 5, 3))

    true_result = np.zeros((len(all_countries), 3))

    all_athletes = []

    if year != '2028':
        for k, country in enumerate(all_countries):
            true_result[k, :] = country_data[year][country][0:3]
            all_athletes.extend(athlete_data[year][country].keys())
    elif '2024' in total_years:
        for k, country in enumerate(all_countries):
            true_result[k, :] = country_data['2024'][country][0:3]
            all_athletes.extend(athlete_data['2024'][country].keys())

    athlete_record = np.zeros((len(all_athletes), 5, 3))
    athlete_to_country = np.zeros((len(all_countries), len(all_athletes)))
    host_country_index = None

    for k, country in enumerate(all_countries):
        if country == host_history[year]:
            host_country_index = k
            
        for t, athlete in enumerate(all_athletes):
            if year != '2028':
                if athlete in athlete_data[year][country].keys():
                    athlete_to_country[k, t] = 1
            
            elif '2024' in total_years:
                if athlete in athlete_data['2024'][country].keys():
                    athlete_to_country[k, t] = 1

    if year != '2028':
        for j in range(1, 6):
            for k, country in enumerate(all_countries):
                if country in country_data[total_years[i-j]].keys():
                    country_record[k, j-1, :] = country_data[total_years[i-j]][country][0:3]

                    for t, athlete in enumerate(all_athletes):
                        if athlete in athlete_data[total_years[i-j]][country].keys():
                            athlete_record[t, j-1, :] = athlete_data[total_years[i-j]][country][athlete][0:3]
    
    elif '2024' in total_years:
        for j in range(1, 6):
            for k, country in enumerate(all_countries):
                if country in country_data[total_years[i-j]].keys():
                    country_record[k, j-1, :] = country_data[total_years[i-j]][country][0:3]

                    for t, athlete in enumerate(all_athletes):
                        if athlete in athlete_data[total_years[i-j]][country].keys():
                            athlete_record[t, j-1, :] = athlete_data[total_years[i-j]][country][athlete][0:3]

    country_record = torch.from_numpy(country_record)
    athlete_record = torch.from_numpy(athlete_record)
    athlete_to_country = torch.from_numpy(athlete_to_country)
    true_result = torch.from_numpy(true_result)

    if year != '2028':
        data_list.append({'country_record': country_record,
                          'athlete_record': athlete_record,
                          'athlete_to_country': athlete_to_country,
                          'host_country_index': host_country_index,
                          'true_result': true_result})
    
    test_dict[year] = {'country_record': country_record,
                       'athlete_record': athlete_record,
                       'athlete_to_country': athlete_to_country,
                       'host_country_index': host_country_index,
                       'true_result': true_result,
                       'all_countries': all_countries}

import csv
with open('tensor.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(test_dict['2028']['country_record'].numpy())

# Settings of model, loss function and optimizer
model = MedalPredictionModel(m_years)
criterion = nn.MSELoss() # Example loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
#soft_norm = nn.Softmax(dim=0)

# Number of epochs to train the model
epochs = 1000

def train(epochs, model, criterion, optimizer, data_list):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0  # To track the loss for the epoch

        # Loop over the dataset (one data point at a time)
        for i in range(len(data_list)):  # Iterate over the number of countries (or samples)
            # For each sample (you might want to get data for one country, one athlete batch)
            current_country_data = data_list[i]['country_record'].float()
            current_athlete_data = data_list[i]['athlete_record'].float()
            current_athlete_to_country = data_list[i]['athlete_to_country'].float()
            current_host_country_index = data_list[i]['host_country_index']
            current_true_result = data_list[i]['true_result'].float()

            # Forward pass for the current sample
            output = model(current_country_data, current_athlete_data, current_athlete_to_country, current_host_country_index)

            loss = criterion(output, current_true_result)
            weights = torch.where(current_true_result != 0, current_true_result ** 2 + 1, torch.tensor(1.0))
            weighted_loss = (loss * weights).mean()

            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            weighted_loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            total_loss += weighted_loss.item()  # Accumulate loss for this batch

            if i == 0:
                print(f'Predicted probabilities is: {output}')
                print(f'Target probabilities is: {current_true_result}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_list)}')
        
        checkpoint_path = f'model_epoch_{epoch+1}.pth'
        if epoch == 0 or (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(data_list),
            }, checkpoint_path)

def validation(model, criterion, data_dict):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation
        # Extract data for current batch (assuming data_dict is a list of validation samples)
        current_country_data = data_dict['country_record'].float()
        current_athlete_data = data_dict['athlete_record'].float()
        current_athlete_to_country = data_dict['athlete_to_country'].float()
        current_host_country_index = data_dict['host_country_index']
        current_true_result = data_dict['true_result'].float()

        # Forward pass for the current batch
        output = model(current_country_data, current_athlete_data, current_athlete_to_country, current_host_country_index)
        
        loss = criterion(output, current_true_result)

        total_loss += loss.item()
        
    avg_loss = total_loss / len(data_dict)

    print(f'Validation Loss: {avg_loss:.4f}')

    return output.numpy()

def inference(model, data_dict):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        # Extract data for current batch (assuming data_dict is a list of validation samples)
        current_country_data = data_dict['country_record'].float()
        current_athlete_data = data_dict['athlete_record'].float()
        current_athlete_to_country = data_dict['athlete_to_country'].float()
        current_host_country_index = data_dict['host_country_index']
        current_true_result = data_dict['true_result'].float()

        # Forward pass for the current batch
        output = model(current_country_data, current_athlete_data, current_athlete_to_country, current_host_country_index)

    return output.numpy()

train(epochs, model, criterion, optimizer, data_list)

model.load_state_dict(torch.load('model_epoch_1000.pth')['model_state_dict'])

country_power = {}
for year, test_data in test_dict.items():
    out_matrix= validation(model, criterion, test_data)
    all_countries = test_data['all_countries']
    country_power[year] = {}
    country_power[year]['true'] = {}
    country_power[year]['infer'] = {}
    for j, country in enumerate(all_countries):
        country_power[year]['infer'][country] = out_matrix[j].tolist()
        country_power[year]['true'][country] = test_data['true_result'][j].tolist()

with open('result_infer.json', 'w') as json_file:
    json.dump(country_power, json_file, indent=4)

import matplotlib.pyplot as plt
def visualize_weights(model):
    for name, param in model.named_parameters():
        if isinstance(getattr(model, name.split('.')[0]), nn.LayerNorm):
            continue
        if 'weight' in name:
            plt.figure(figsize=(5, 5))
            plt.title(f'{name} - Weights/Biases')
            plt.imshow(param.detach().numpy(), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.show()

visualize_weights(model)