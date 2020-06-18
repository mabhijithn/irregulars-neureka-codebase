# Evaluate

Code to produce seizure detection and a hypothesis file.

When evaluation a new file:

1. Save a filtered version of the data using ICLabel filtering
2. Make prediction files using the U-Net model
3. Fuse the U-Net predictions with the LSTM model
4. Use the predictions to produce a hypothesis file