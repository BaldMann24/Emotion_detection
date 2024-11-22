import serial
import time
import csv
from datetime import datetime
from scipy.signal import butter, filtfilt
import numpy as np

# Configure the serial port (make sure the port matches your setup)
ser = serial.Serial('COM10', 9600)

# Give some time to establish the serial connection
time.sleep(2)

print("Reading live EEG values from AF7 and TP9...")

# Band-pass filter design
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def normalize(data, new_min=-100, new_max=100):
    """Normalize the data to the specified range."""
    old_min = np.min(data)
    old_max = np.max(data)
    normalized = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return normalized

# Filter parameters
fs = 256 
lowcut = 3.0  
highcut = 30.0  


with open('subjecta-neutral-1.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

  
    csv_writer.writerow(['Timestamp', 'AF7 Value', 'TP9 Value'])  

  
    af7_values = []
    tp9_values = []

    
    for _ in range(10000):
        try:
            # Read data from the serial port
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip() 
                values = line.split(",")  

                # Convert string values to float and append to lists
                af7_value = float(values[0])
                tp9_value = float(values[1])
                af7_values.append(af7_value)
                tp9_values.append(tp9_value)

                # Only filter after collecting enough samples
                if len(af7_values) >= fs:
                    # Apply the band-pass filter
                    filtered_af7 = bandpass_filter(af7_values, lowcut, highcut, fs)
                    filtered_tp9 = bandpass_filter(tp9_values, lowcut, highcut, fs)

                    # Normalize the last filtered values
                    normalized_af7 = normalize(filtered_af7)
                    normalized_tp9 = normalize(filtered_tp9)

                    # Get the current timestamp
                    timestamp = datetime.now().timestamp()  # Get Unix timestamp

                    # Write the timestamp and normalized values to the CSV file
                    csv_writer.writerow([timestamp] + [normalized_af7[-1], normalized_tp9[-1]])  # Take the last normalized value
                    print(f"{timestamp:.3f} - AF7: {normalized_af7[-1]:.3f}, TP9: {normalized_tp9[-1]:.3f}")  # Print the live values with timestamp

                    # Remove oldest value to maintain the buffer size
                    af7_values.pop(0)
                    tp9_values.pop(0)

            time.sleep(0.001)  # Wait for 3 milliseconds between readings
        except Exception as e:
            print(f"Error: {e}")
            break

print("Data collection complete.")
ser.close()  # Close the serial port when done