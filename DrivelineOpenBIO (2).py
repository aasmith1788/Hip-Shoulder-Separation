#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('conda install -c conda-forge ezc3d -y')



# In[1]:


import ezc3d
import pandas as pd
import numpy as np
import os

def c3d_to_dataframe(file_path):
    c = ezc3d.c3d(file_path)
    
    # Extract point data
    point_data = c['data']['points']
    point_labels = c['parameters']['POINT']['LABELS']['value']
    
    # Reshape the data
    # From (4, num_markers, num_frames) to (num_frames, num_markers * 3)
    reshaped_data = point_data[:3, :, :].transpose(2, 1, 0).reshape(-1, point_data.shape[1] * 3)
    
    # Create column names
    columns = [f"{label}_{axis}" for label in point_labels for axis in ['X', 'Y', 'Z']]
    
    # Create DataFrame
    df = pd.DataFrame(reshaped_data, columns=columns)
    
    # Add a time column
    frame_rate = c['header']['points']['frame_rate']
    df['Time'] = np.arange(len(df)) / frame_rate
    
    # Extracting information from filename
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    if len(parts) >= 7:  # Check if the filename follows expected format
        df['USERid'] = parts[0]
        df['SESSIONid'] = parts[1]
        df['HEIGHT'] = parts[2]
        df['WEIGHT'] = parts[3]
        df['PITCHNUMBER'] = parts[4]
        df['PITCHTYPE'] = parts[5]
        speed_str = parts[6].replace('.c3d', '')  # Remove file extension and adjust format
        df['PITCHSPEED'] = float(speed_str[:-1] + '.' + speed_str[-1]) if speed_str[:-1].isdigit() else None
    
    df['Filename'] = filename
    
    return df

# Directory containing your C3D files
main_directory = "C:/Users/aasmi/OneDrive/Documents/GitHub/openbiomechanics/baseball_pitching/data/c3d"

all_data = []

for subfolder in os.listdir(main_directory):
    subfolder_path = os.path.join(main_directory, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.c3d'):
                file_path = os.path.join(subfolder_path, filename)
                df = c3d_to_dataframe(file_path)
                all_data.append(df)

# Combine all DataFrames
combined_df = pd.concat(all_data, ignore_index=True)

# Display information about the combined DataFrame
print(combined_df.info())

# Display the first few rows
print(combined_df.head())

# Save to CSV (optional)
combined_df.to_csv('combined_c3d_data.csv', index=False)




# In[2]:


import numpy as np
import pandas as pd

def calculate_angle(x, y):
    return np.arctan2(y, x) * 180 / np.pi

def calculate_angle_difference(angle1, angle2):
    # Direct subtraction to preserve the sign
    diff = angle1 - angle2
    
    # Adjust to stay within the -180 to 180 range
    diff = (diff + 180) % 360 - 180
    
    return diff

# Calculate Pelvis and Shoulder vectors
pelvis_vector_x = combined_df['RASI_X'] - combined_df['LASI_X']
pelvis_vector_y = combined_df['RASI_Y'] - combined_df['LASI_Y']
shoulder_vector_x = combined_df['RSHO_X'] - combined_df['LSHO_X']
shoulder_vector_y = combined_df['RSHO_Y'] - combined_df['LSHO_Y']

# Calculate angles
pelvis_angle = calculate_angle(pelvis_vector_x, pelvis_vector_y)
shoulder_angle = calculate_angle(shoulder_vector_x, shoulder_vector_y)

# Calculate Hip-Shoulder Separation
hip_shoulder_separation = calculate_angle_difference(pelvis_angle, shoulder_angle)

# Create a new DataFrame with all the new columns at once
new_columns_df = pd.DataFrame({
    'Pelvis_Vector_X': pelvis_vector_x,
    'Pelvis_Vector_Y': pelvis_vector_y,
    'Shoulder_Vector_X': shoulder_vector_x,
    'Shoulder_Vector_Y': shoulder_vector_y,
    'Pelvis_Angle': pelvis_angle,
    'Shoulder_Angle': shoulder_angle,
    'Hip_Shoulder_Separation': hip_shoulder_separation
})

# Concatenate the new columns back to the original DataFrame
combined_df = pd.concat([combined_df, new_columns_df], axis=1)

# Now, combined_df contains all the original columns plus the new calculated columns
print(combined_df.head())



# In[3]:


# Check the number of NaN values in the 'Hip_Shoulder_Separation' column
na_count = combined_df['Hip_Shoulder_Separation'].isna().sum()

# Print the result
print(f"Number of NaN values in 'Hip_Shoulder_Separation': {na_count}")

# Drop rows where 'Hip_Shoulder_Separation' is NaN
combined_df = combined_df.dropna(subset=['Hip_Shoulder_Separation'])

# Check the number of rows after dropping NaN values
print(f"Number of rows after dropping NaN values: {len(combined_df)}")


# In[4]:


# Group by Filename and find the hip shoulder separation value furthest from 0
grouped = combined_df.groupby('Filename')['Hip_Shoulder_Separation'].agg(lambda x: x.loc[x.abs().idxmax()])

# Create a new dataframe with the results
result_df = pd.DataFrame(grouped).reset_index()

# Create the p_throws column based on the sign of the hip shoulder separation
result_df['p_throws'] = np.where(result_df['Hip_Shoulder_Separation'] < 0, 'L', 'R')

# Merge the result back to the original dataframe
combined_df = combined_df.merge(result_df[['Filename', 'p_throws']], on='Filename', how='left')


# In[5]:


# Define a function to calculate hand velocity based on throwing hand
def calculate_hand_velocity(row, prev_row):
    if row['p_throws'] == 'R':
        return np.sqrt((row['RWRA_X'] - prev_row['RWRA_X'])**2 + 
                       (row['RWRA_Y'] - prev_row['RWRA_Y'])**2 + 
                       (row['RWRA_Z'] - prev_row['RWRA_Z'])**2)
    elif row['p_throws'] == 'L':
        return np.sqrt((row['LWRA_X'] - prev_row['LWRA_X'])**2 + 
                       (row['LWRA_Y'] - prev_row['LWRA_Y'])**2 + 
                       (row['LWRA_Z'] - prev_row['LWRA_Z'])**2)
    else:
        return np.nan  # Handle cases where 'p_throws' is not defined

# Initialize a list to store the velocities
hand_velocities = []

# Group by 'Filename' to calculate velocities within each group
for name, group in combined_df.groupby('Filename'):
    # Calculate hand velocity for each row, starting from the second row within each group
    velocities = [np.nan]  # Insert NaN for the first row since it doesn't have a previous row to compare to
    for i in range(1, len(group)):
        current_row = group.iloc[i]
        previous_row = group.iloc[i - 1]
        velocity = calculate_hand_velocity(current_row, previous_row)
        velocities.append(velocity)
    
    # Assign the velocities back to the original DataFrame
    combined_df.loc[group.index, 'Hand_Velocity'] = velocities

# Check the DataFrame to ensure velocities have been added
print(combined_df.head())




# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def furthest_from_zero_with_time(group):
    pos_max = group['Hip_Shoulder_Separation'].max()
    neg_min = group['Hip_Shoulder_Separation'].min()
    
    if abs(pos_max) > abs(neg_min):
        idx = group['Hip_Shoulder_Separation'].idxmax()
    else:
        idx = group['Hip_Shoulder_Separation'].idxmin()
    
    return pd.Series({
        'max_hip_shoulder_separation': group.loc[idx, 'Hip_Shoulder_Separation'],
        'time_at_max_hip_shoulder_separation': group.loc[idx, 'Time']
    })

# Group by Filename and find the row with the highest Hand_Velocity
result_df = combined_df.loc[combined_df.groupby('Filename')['Hand_Velocity'].idxmax()]

# Get the max_hip_shoulder_separation (furthest from 0) and its time for each Filename
max_hip_shoulder_data = combined_df.groupby('Filename').apply(furthest_from_zero_with_time)

# Select the columns we need
result_df = result_df[['Filename', 'Hand_Velocity', 'Time', 'Hip_Shoulder_Separation', 'p_throws']]

# Rename the Hip_Shoulder_Separation column to hip_shoulder_separation_at_br
result_df = result_df.rename(columns={'Hip_Shoulder_Separation': 'hip_shoulder_separation_at_br'})

# Add the max_hip_shoulder_separation and its time to the result_df
result_df = result_df.merge(max_hip_shoulder_data, left_on='Filename', right_index=True)

# Rename the Time column to time_at_max_hand_velocity for clarity
result_df = result_df.rename(columns={'Time': 'time_at_max_hand_velocity'})

# Calculate the angular velocity (degrees per second)
result_df['angular_velocity'] = abs(result_df['hip_shoulder_separation_at_br'] - result_df['max_hip_shoulder_separation']) / (result_df['time_at_max_hand_velocity'] - result_df['time_at_max_hip_shoulder_separation'])

# Remove rows where Filename ends with 'model.c3d'
result_df = result_df[~result_df['Filename'].str.endswith('model.c3d')]

# Extract features from Filename
result_df['USERid'] = result_df['Filename'].str.split('_').str[0]
result_df['SESSIONid'] = result_df['Filename'].str.split('_').str[1]
result_df['HEIGHT'] = result_df['Filename'].str.split('_').str[2].astype(float)
result_df['WEIGHT'] = result_df['Filename'].str.split('_').str[3].astype(float)
result_df['PITCHNUMBER'] = result_df['Filename'].str.split('_').str[4].astype(int)
result_df['PITCHTYPE'] = result_df['Filename'].str.split('_').str[5]
result_df['PITCHSPEED'] = result_df['Filename'].str.split('_').str[6].str.split('.').str[0].astype(float) / 10

# Create the new session_pitch variable
result_df['session_pitch'] = result_df['SESSIONid'].str[-4:] + '_' + result_df['PITCHNUMBER'].astype(str)

# Reset the index for cleaner output
result_df = result_df.reset_index(drop=True)

# Filter out rows where 'angular_velocity' is below 0 or greater than 700
result_df = result_df[(result_df['angular_velocity'] >= 0)]

# Display the first few rows of the resulting DataFrame
print("First few rows of the resulting DataFrame:")
print(result_df.head())

# Display basic information about the DataFrame
print("\nDataFrame Info:")
print(result_df.info())

# Display summary statistics of the numerical columns
print("\nSummary Statistics:")
print(result_df.describe())

# Display the count of different values in p_throws and PITCHTYPE
print("\nCounts of p_throws:")
print(result_df['p_throws'].value_counts())
print("\nCounts of PITCHTYPE:")
print(result_df['PITCHTYPE'].value_counts())

# Display some rows to compare angular velocity and session_pitch
print("\nAngular velocity and session_pitch samples:")
print(result_df[['Filename', 'p_throws', 'max_hip_shoulder_separation', 'hip_shoulder_separation_at_br', 'angular_velocity', 'session_pitch']].head(10))

# Calculate correlations
numerical_columns = result_df.select_dtypes(include=[np.number]).columns
correlation_matrix = result_df[numerical_columns].corr()

# Extract correlations with PITCHSPEED
pitchspeed_correlations = correlation_matrix['PITCHSPEED'].sort_values(ascending=False)

print("\nCorrelations with PITCHSPEED:")
print(pitchspeed_correlations)

# Create a bar plot of correlations with PITCHSPEED
plt.figure(figsize=(12, 6))
pitchspeed_correlations.drop('PITCHSPEED').plot(kind='bar')
plt.title('Correlations with PITCHSPEED')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Scatter plot of Angular Velocity vs Pitch Speed with regression line
plt.figure(figsize=(8, 4))
sns.scatterplot(x='angular_velocity', y='PITCHSPEED', hue='p_throws', data=result_df)

# Calculate the least squares regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(result_df['angular_velocity'], result_df['PITCHSPEED'])
line = slope * result_df['angular_velocity'] + intercept

# Plot the regression line
plt.plot(result_df['angular_velocity'], line, color='r', label=f'Regression line')

# Calculate R-squared
r_squared = r_value ** 2

# Add correlation and R-squared to the plot
plt.text(0.05, 0.95, f'Correlation: {r_value:.2f}\nR-squared: {r_squared:.2f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.title('Angular Velocity vs Pitch Speed (Angular Velocity)')
plt.xlabel('Angular Velocity (degrees/second)')
plt.ylabel('Pitch Speed (mph)')
plt.legend()
plt.tight_layout()
plt.savefig('av.png', dpi=300, bbox_inches='tight')
plt.show()

# Display sample rows with new session_pitch variable
print("\nSample rows with new session_pitch variable:")
print(result_df[['Filename', 'SESSIONid', 'PITCHNUMBER', 'session_pitch']].head(10))



# In[7]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Function to categorize pitch speeds
def categorize_pitch_speed(speed):
    if speed >= 90:
        return '90+ mph'
    elif 85 <= speed < 90:
        return '85-90 mph'
    elif 80 <= speed < 85:
        return '80-85 mph'
    else:
        return 'Less than 80 mph'
# Apply categorization
result_df['speed_category'] = result_df['PITCHSPEED'].apply(categorize_pitch_speed)
# Function to perform ANOVA and Tukey's HSD
def perform_anova_and_tukey(df, var):
    model = ols(f'{var} ~ C(speed_category)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Perform Tukey's HSD
    tukey = pairwise_tukeyhsd(df[var], df['speed_category'])

    return anova_table, tukey
# Function to plot boxplot for each variable
def plot_boxplot(df, var):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='speed_category', y=var, data=df, 
                order=['Less than 80 mph', '80-85 mph', '85-90 mph', '90+ mph'])
    plt.title(f'{var} across Pitch Speed Categories')
    plt.xlabel('Pitch Speed Category')
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig('angular_velocity_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()


anova_result, tukey_result = perform_anova_and_tukey(result_df, 'angular_velocity')
print(f"\nANOVA results for {'angular_velocity'}:")
print(anova_result)
print(f"\nTukey's HSD results for {'angular_velocity'}:")
print(tukey_result)
plot_boxplot(result_df, 'angular_velocity')




# In[8]:


# Step 1: Filter for PITCHSPEED above 90
filtered_df = result_df[result_df['PITCHSPEED'] > 90]

# Step 2: Find the row with the highest angular velocity
max_angular_velocity_row = filtered_df.loc[filtered_df['angular_velocity'].idxmax()]

# Step 3: Get the Filename
filename_highest_angular_velocity = max_angular_velocity_row['Filename']

print(f"The Filename with the highest angular velocity and PITCHSPEED above 90 is: {filename_highest_angular_velocity}")

# Print additional information about this row
print("\nDetails of the row:")
print(max_angular_velocity_row)


# Step 1: Filter for throws below 80 mph
filtered_df = result_df[result_df['PITCHSPEED'] < 80]

# Step 2: Find the row with the lowest angular velocity
min_angular_velocity_row = filtered_df.loc[filtered_df['angular_velocity'].idxmin()]

# Step 3: Get the Filename
filename_lowest_angular_velocity = min_angular_velocity_row['Filename']

print(f"The Filename with the slowest angular velocity and PITCHSPEED below 80 is: {filename_lowest_angular_velocity}")

# Print additional information about this row
print("\nDetails of the row:")
print(min_angular_velocity_row)


# In[9]:


# Use a basic style
plt.style.use('default')

# Filter the combined_df for the specific filename
file_df = combined_df[combined_df['Filename'] == '001466_002653_75_170_001_FF_909.c3d']

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the data with a thicker line in red
ax.plot(file_df['Time'], file_df['Hip_Shoulder_Separation'], linewidth=2, color='red', label='Hip-Shoulder Separation')

# Add vertical lines at specified times with updated labels
ax.axvline(x=1.341667, color='gray', linestyle='-', linewidth=1, label='Max Hip-Shoulder Separation')
ax.axvline(x=1.427778, color='red', linestyle='-', linewidth=1, label='Ball Release')

# Customize the plot
ax.set_title('Hip-Shoulder Separation over Time (90.9 mph RHP)', fontsize=16, fontweight='bold')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Hip-Shoulder Separation', fontsize=12)

# Ensure y-axis starts at 0
ax.set_ylim(bottom=0, top=75)
ax.set_xlim(0, 1.6)  # Extended to 1.6 seconds

# Add black grid on white background
ax.grid(True, linestyle='-', color='black', alpha=0.2)
ax.set_facecolor('white')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=10)

# Add x-axis labels
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_xticklabels(['0s', '0.2s', '0.4s', '0.6s', '0.8s', '1s', '1.2s', '1.4s', '1.6s'])

# Add legend in the top left corner
ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.01, 0.99), framealpha=0.8)

# Tight layout to use space efficiently
plt.tight_layout()

plt.savefig('hip_shoulder_separation_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[10]:


# Use a basic style
plt.style.use('default')

# Filter the combined_df for the specific filename
file_df = combined_df[combined_df['Filename'] == '001027_001782_72_215_002_FF_738.c3d']

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))  # Increased figure size to accommodate legend

# Plot the data with a thicker line in red
ax.plot(file_df['Time'], file_df['Hip_Shoulder_Separation'], linewidth=2, color='red', label='Hip-Shoulder Separation')

# Add vertical lines at specified times with updated labels
ax.axvline(x=1.427778, color='gray', linestyle='-', linewidth=1, label='Max Hip-Shoulder Separation')
ax.axvline(x=1.594444, color='red', linestyle='-', linewidth=1, label='Ball Release')

# Customize the plot
ax.set_title('Hip-Shoulder Separation over Time (73.8 mph LHP)', fontsize=16, fontweight='bold')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Hip-Shoulder Separation', fontsize=12)

# Set axis limits
ax.set_ylim(-50, 50)  # Y-axis from -50 to 50
ax.set_xlim(0, 2)  # X-axis from 0 to 2 seconds

# Add black grid on white background
ax.grid(True, linestyle='-', color='black', alpha=0.2)
ax.set_facecolor('white')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=10)

# Add x-axis labels
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2])
ax.set_xticklabels(['0s', '0.2s', '0.4s', '0.6s', '0.8s', '1s', '1.2s', '1.4s', '1.6s', '1.8s', '2s'])

# Set y-axis ticks
ax.set_yticks(range(-50, 51, 25))

# Add legend in the top left corner
ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.01, 0.99), framealpha=0.8)

# Tight layout to use space efficiently
plt.tight_layout()

plt.savefig('hip_shoulder_separation_plot738.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


# In[11]:


# Load the dataset
file_path = 'C:\\Users\\aasmi\\OneDrive\\Documents\\GitHub\\openbiomechanics\\baseball_pitching\\data\\poi\\poi_metrics.csv'
poi_metrics = pd.read_csv(file_path)
poi_metrics = poi_metrics.dropna()
poi_metrics

# Create the poi_metrics DataFrame
poi_metrics = poi_metrics[['session_pitch', 'max_rotation_hip_shoulder_separation']]

complete = result_df.merge(poi_metrics, on='session_pitch', how='inner')


# In[12]:


complete = complete.dropna(subset=['max_rotation_hip_shoulder_separation'])

# Convert max_rotation_hip_shoulder_separation to absolute values
complete['max_rotation_hip_shoulder_separation'] = complete['max_rotation_hip_shoulder_separation'].abs()

# Convert max_hip_shoulder_separation to absolute values (if it still exists in the DataFrame)
if 'max_hip_shoulder_separation' in complete.columns:
    complete['max_hip_shoulder_separation'] = complete['max_hip_shoulder_separation'].abs()



# Assuming 'complete' is your dataframe
# Calculate the Pearson correlation coefficient
correlation, p_value = stats.pearsonr(complete['max_rotation_hip_shoulder_separation'], 
                                      complete['max_hip_shoulder_separation'])

print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
if p_value < 0.05:
    print("The correlation is statistically significant (p < 0.05).")
else:
    print("The correlation is not statistically significant (p >= 0.05).")

if abs(correlation) < 0.3:
    strength = "weak"
elif abs(correlation) < 0.7:
    strength = "moderate"
else:
    strength = "strong"

print(f"The correlation is {strength} ({correlation:.4f}).")


# In[ ]:




