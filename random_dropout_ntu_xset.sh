#!/bin/bash

# Define the list of values for miss_amount
miss_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)


work_dir="./results/ntu120/demo"
echo " "
config_file="./config/nturgbd120-cross-set/rand_scale.yaml"

# Print the current working directory and check if the config file exists
echo "Current working directory: $(pwd)"

if [ ! -f "$config_file" ]; then
    echo "Error: Config file does not exist at $config_file"
    exit 1
else
    echo "Config file exists at $config_file"
fi

# Loop over each value
for miss_value in "${miss_values[@]}"
do
    echo "Running script with miss_amount: $miss_value"
    
    # Update the YAML config file with the new miss_amount value using sed
    #sed -i "s/miss_amount: [0-9]\+/miss_amount: $miss_value/" "$config_file"
    sed -i "s/miss_amount: [0-9]*\.[0-9]\+/miss_amount: $miss_value/" "$config_file"

    # Check if sed updated the file correctly
    updated_value=$(grep 'miss_amount' "$config_file" | awk '{print $2}')
    echo "miss value $miss_value updated value $updated_value"
    if [ "$updated_value" != "$miss_value" ]; then
        echo "Error: Failed to update miss_amount to $miss_value in $config_file"
        exit 1
    else
        echo "Successfully updated miss_amount to $miss_value in $config_file"
    fi
    
    # Print the updated config file content
    echo "Updated config file content:"
    grep 'miss_amount' "$config_file"

    python3 main.py --config "$config_file" --work-dir "$work_dir" --phase test --save-score True --weights "./results/ntu120/xset/FR-Head_Joint_87.32/runs.pt" --device 0

    echo "Finished running script with miss_amount: $miss_value"
    echo "---------------------------------------------"
done