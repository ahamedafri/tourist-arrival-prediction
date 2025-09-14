import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import glob

# Function to merge CSV files
def merge_csv_files():
    # Get all CSV files in the current directory
    csv_files = sorted(glob.glob('*.csv'))  # Sort to ensure chronological order
    
    # Create empty list to store dataframes
    dfs = []
    
    # Read each CSV file and add year column
    for file in csv_files:
        year = int(file.split('.')[0])  # Extract year from filename
        df = pd.read_csv(file)
        
        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Convert visitor numbers by removing commas
        numeric_columns = [col for col in df.columns if col != 'Country']
        for col in numeric_columns:
            # Remove commas and convert to numeric, replacing errors with NaN
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Fill NaN values with the mean of the column
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        df['Year'] = year
        dfs.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

# Function to prepare data for LSTM
def prepare_data(df, sequence_length=12):
    # Melt the dataframe to convert months to rows
    df_melted = df.melt(id_vars=['Country', 'Year'], 
                        var_name='Month', 
                        value_name='Visitors')

    # Create proper month numbers for datetime
    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    df_melted['Month_Num'] = df_melted['Month'].map(month_map)

    # Create datetime index
    df_melted['Date'] = pd.to_datetime(
        df_melted['Year'].astype(str) + '-' + df_melted['Month_Num']
    )

    # Sort by country and date
    df_melted = df_melted.sort_values(['Country', 'Date'])

    sequences = []
    targets = []
    scalers = {}
    country_indices = []

    # Prepare sequences for each country
    for country in df_melted['Country'].unique():
        country_data = df_melted[df_melted['Country'] == country]['Visitors'].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(country_data.reshape(-1, 1))
        scalers[country] = scaler
        # Create sequences
        for i in range(len(scaled_data) - sequence_length):
            sequences.append(scaled_data[i:(i + sequence_length)])
            targets.append(scaled_data[i + sequence_length])
            country_indices.append(country)

    return np.array(sequences), np.array(targets), scalers, country_indices

# Main execution
if __name__ == "__main__":
    # Merge CSV files
    print("Merging CSV files...")
    merged_data = merge_csv_files()
    
    # Prepare data for LSTM
    print("Preparing data for LSTM...")
    X, y, scaler = prepare_data(merged_data)
    
    # Print dataset shape
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(12, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")
    
    # Save the model
    print("\nSaving model...")
    model.save('tourism_lstm_model.h5')
    print("Model saved as 'tourism_lstm_model.h5'")
