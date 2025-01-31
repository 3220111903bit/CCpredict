import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MedalPredictionModel(nn.Module):
    def __init__(self, m_years):
        super(MedalPredictionModel, self).__init__()

        self.m_years = m_years

        # Define the learnable parameters
        self.W_A = nn.Parameter(torch.randn(m_years*3, 6))  # Country transformation
        self.W_B = nn.Parameter(torch.randn(m_years*3, 4))  # Athlete transformation
        self.host_effect = nn.Parameter(torch.randn(10))  # Host effect

        # Define the self-attention layer (in this case, for attention between countries)
        self.attn = nn.MultiheadAttention(embed_dim=10, num_heads=1, batch_first=True)  # Attention between countries
        
        # Define the final transformation matrix (for output (c, 1, 3))
        self.final_matrix = nn.Parameter(torch.randn(10, 1))  # Final transformation
        self.softmax = nn.Softmax(dim=0)

    def forward(self, country_data, athlete_data, athlete_to_country, host_country_index):
        a, c, m = athlete_data.shape[0], country_data.shape[0], country_data.shape[1]
        # country_data: Tensor with shape (c, m, 3) - where "c" is total countries
        # athlete_data: Tensor with shape (a, m, 3) - where "a" is total athletes, "m" is years, 5 features per athlete
        # athlete_to_country: Shape (c, a), mapping athletes to countries (only contain 0 and 1)
        # host_country_index: Shape (c, 1), Index of the host country

        # Transform country data using W_A (Country transformation)
        country_data = F.sigmoid(country_data.reshape(c, -1))
        country_data = torch.matmul(country_data, self.W_A) # shape (c, 6)
        country_data = country_data

        # Apply transformation to athlete data using W_B
        athlete_data = F.sigmoid(athlete_data.reshape(a, -1))
        athlete_data = torch.matmul(athlete_data, self.W_B) # shape (a, 4)
        athlete_data = athlete_data

        # Now we need to add athlete data to the corresponding countries using the athlete_to_country matrix
        # athlete_to_country: Shape (c, a), where each column corresponds to an athlete and indicates which country they belong to
        country_athlete_contribution = torch.matmul(athlete_to_country.float(), athlete_data) # shape (c, 4)

        # Combine country data with the athlete contributions
        combined_data = torch.cat((country_data, country_athlete_contribution), dim=1) # shape (c, 10)

        # Add the host effect
        if host_country_index:
            combined_data[host_country_index] += self.host_effect  # Add host effect to the host country

        # Apply activation (ReLU) across the years (m) and medals (3), but not across countries (c)
        activated_data = F.leaky_relu(combined_data, negative_slope=0.1)

        # Step 1: Apply self-attention on the (c, m_years) matrix for the current medal type
        attn_output, _ = self.attn(activated_data, activated_data, activated_data)  # Shape: (c, m_years*3)

        # Now, `attn_output` has shape (c, m_years, 3), which incorporates country-country interactions for each medal type
        attn_output = attn_output + activated_data

        # Apply activation function (ReLU)
        activated_output = F.leaky_relu(attn_output, negative_slope=0.1)  # Shape: (c, m*3)

        # Multiply with the final matrix to get the output
        final_output = torch.matmul(activated_output, self.final_matrix)  # Shape: (c, 3)
        #final_output = self.softmax(final_output)

        return final_output

m_years = 5

import json
import numpy as np

with open('olympic_medals_noteam.json', 'r') as file:
    country_data = json.load(file)
# sport -> year -> country -> medal

with open('athlete_medals_noteam.json', 'r') as file:
    athlete_data = json.load(file)
# sport -> year -> country -> athlete -> medal

with open('host_history.json', 'r') as file:
    host_history = json.load(file)

assert set(country_data.keys()) == set(athlete_data.keys())

data_list = []
test_list = []
infer_list = []

for sport in sorted(country_data.keys()):
    country_data_sport = country_data[sport]
    athlete_data_sport = athlete_data[sport]

    assert set(country_data_sport.keys()) == set(country_data_sport.keys())

    total_years = sorted(country_data_sport.keys())

    for i, year in enumerate(total_years + ['2028']):
        if i < 5:
            continue

        if year != '2028':
            all_countries = sorted(country_data_sport[year].keys())
        elif '2024' in total_years:
            all_countries = sorted(country_data_sport['2024'].keys())

        country_record = np.zeros((len(all_countries), 5, 3))

        true_result = np.zeros((len(all_countries), 3))
        athlete_number = np.zeros(len(all_countries))

        all_athletes = []

        if year != '2028':
            for k, country in enumerate(all_countries):
                true_result[k, :] = country_data_sport[year][country][0:3]
                athlete_number[k] = country_data_sport[year][country][4]
                all_athletes.extend(athlete_data_sport[year][country].keys())
        elif '2024' in total_years:
            for k, country in enumerate(all_countries):
                true_result[k, :] = country_data_sport['2024'][country][0:3]
                athlete_number[k] = country_data_sport['2024'][country][4]
                all_athletes.extend(athlete_data_sport['2024'][country].keys())

        athlete_record = np.zeros((len(all_athletes), 5, 3))
        athlete_to_country = np.zeros((len(all_countries), len(all_athletes)))
        host_country_index = None

        for k, country in enumerate(all_countries):
            if country == host_history[year]:
                host_country_index = k
                
            for t, athlete in enumerate(all_athletes):
                if year != '2028':
                    if athlete in athlete_data_sport[year][country].keys():
                        athlete_to_country[k, t] = 1
                
                elif '2024' in total_years:
                    if athlete in athlete_data_sport['2024'][country].keys():
                        athlete_to_country[k, t] = 1

        if year != '2028':
            for j in range(1, 6):
                
                for k, country in enumerate(all_countries):
                    if country in country_data_sport[total_years[i-j]].keys():
                        country_record[k, j-1, :] = country_data_sport[total_years[i-j]][country][0:3]

                        for t, athlete in enumerate(all_athletes):
                            if athlete in athlete_data_sport[total_years[i-j]][country].keys():
                                athlete_record[t, j-1, :] = athlete_data_sport[total_years[i-j]][country][athlete][0:3]
        
        elif '2024' in total_years:
            for j in range(1, 6):
                
                for k, country in enumerate(all_countries):
                    if country in country_data_sport[total_years[i-j]].keys():
                        country_record[k, j-1, :] = country_data_sport[total_years[i-j]][country][0:3]

                        for t, athlete in enumerate(all_athletes):
                            if athlete in athlete_data_sport[total_years[i-j]][country].keys():
                                athlete_record[t, j-1, :] = athlete_data_sport[total_years[i-j]][country][athlete][0:3]

        country_record = torch.from_numpy(country_record)
        athlete_record = torch.from_numpy(athlete_record)
        athlete_to_country = torch.from_numpy(athlete_to_country)
        true_result = torch.from_numpy(true_result)
        athlete_number = torch.from_numpy(athlete_number)

        if year != '2028':
            data_list.append({'country_record': country_record,
                            'athlete_record': athlete_record,
                            'athlete_to_country': athlete_to_country,
                            'host_country_index': host_country_index,
                            'true_result': true_result})
        
        if year == '2024':
            test_list.append({'country_record': country_record,
                              'athlete_record': athlete_record,
                              'athlete_to_country': athlete_to_country,
                              'host_country_index': host_country_index,
                              'true_result': true_result,
                              'all_countries': all_countries,
                              'athlete_number': athlete_number})
            
        if year == '2028' and '2024' in total_years:
            infer_list.append({'country_record': country_record,
                              'athlete_record': athlete_record,
                              'athlete_to_country': athlete_to_country,
                              'host_country_index': host_country_index,
                              'true_result': true_result,
                              'all_countries': all_countries,
                              'athlete_number': athlete_number})

# Settings of model, loss function and optimizer
model = MedalPredictionModel(m_years)
criterion = nn.MSELoss() # Example loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
transformation_matrix = torch.tensor([1, 0.8, 0.6]).reshape(3, 1)
#soft_norm = nn.Softmax(dim=0)

# Number of epochs to train the model
epochs = 200

def train(epochs, model, criterion, optimizer, data_list):
    loss_history = []
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

            current_true_result = torch.matmul(current_true_result, transformation_matrix) + 0.05
            loss = criterion(output, current_true_result)

            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate loss for this batch

            if i == 0:
                print(f'Predicted probabilities is: {output}')
                print(f'Target probabilities is: {current_true_result}')

        avg_loss = total_loss / len(data_list)
        loss_history.append(avg_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_list)}')
        
        checkpoint_path = f'model_epoch_{epoch+1}.pth'
        if epoch == 0 or (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(data_list),
            }, checkpoint_path)

    plt.plot(range(1, epochs + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig('model_loss.png', dpi=300)

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
        current_athlete_number = data_dict['athlete_number']

        scale_factor = (current_true_result[:, 0] * current_athlete_number).sum().item()

        # Forward pass for the current batch
        output = model(current_country_data, current_athlete_data, current_athlete_to_country, current_host_country_index)
        
        current_true_result = torch.matmul(current_true_result, transformation_matrix) + 0.05

        # Calculate loss
        loss = criterion(output, current_true_result)
        total_loss += loss.item()
        

    avg_loss = total_loss / len(data_dict)

    print(f'Validation Loss: {avg_loss:.4f}')

    return output.numpy(), scale_factor

#train(epochs, model, criterion, optimizer, data_list)

def inference(model, data_dict):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        # Extract data for current batch (assuming data_dict is a list of validation samples)
        current_country_data = data_dict['country_record'].float()
        current_athlete_data = data_dict['athlete_record'].float()
        current_athlete_to_country = data_dict['athlete_to_country'].float()
        current_host_country_index = data_dict['host_country_index']
        current_true_result = data_dict['true_result'].float()
        current_athlete_number = data_dict['athlete_number']

        scale_factor = (current_true_result[:, 0] * current_athlete_number).sum().item()

        # Forward pass for the current batch
        output = model(current_country_data, current_athlete_data, current_athlete_to_country, current_host_country_index)

    return output.numpy(), scale_factor

model.load_state_dict(torch.load('model_epoch_200.pth')['model_state_dict'])

country_power = {}
for i in range(len(test_list)):
    out_matrix, scale_factor = validation(model, criterion, test_list[i])
    all_countries = test_list[i]['all_countries']
    for j, country in enumerate(all_countries):
        if country not in country_power:
            country_power[country] = 0
        country_power[country] += out_matrix[j,0] * scale_factor

with open('result_2024.json', 'w') as json_file:
    json.dump(country_power, json_file, indent=4)

country_power = {}
for i in range(len(infer_list)):
    out_matrix, scale_factor = inference(model, infer_list[i])
    all_countries = infer_list[i]['all_countries']
    for j, country in enumerate(all_countries):
        if country not in country_power:
            country_power[country] = 0
        country_power[country] += out_matrix[j,0] * scale_factor

with open('result_2028.json', 'w') as json_file:
    json.dump(country_power, json_file, indent=4)
