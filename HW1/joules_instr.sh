#!/bin/bash

# Record initial energy consumption in microjoules
initial_energy=$(cat /sys/class/powercap/intel-rapl:0/energy_uj)

# Run the program and capture the output (replace ./whetstone with your actual program)
echo "Running the program..."
output=$(./whetstone)

# Extract MIPS value from the program output (assuming it's in the format: "C Converted Double Precision Whetstones: <MIPS>")
mips=$(echo "$output" | grep -oP '(\d+\.\d+) MIPS' | cut -d ' ' -f 1)

# Check if MIPS value was extracted correctly
if [ -z "$mips" ]; then
    echo "Failed to extract MIPS value from program output."
    exit 1
fi

# Extract the duration from the output (assuming the format: "Duration: <duration> sec")
duration=$(echo "$output" | grep -oP 'Duration: \d+ sec' | cut -d ' ' -f 2)

# Check if duration was extracted correctly
if [ -z "$duration" ]; then
    echo "Failed to extract duration from program output."
    exit 1
fi

# Record final energy consumption in microjoules
final_energy=$(cat /sys/class/powercap/intel-rapl:0/energy_uj)

# Calculate energy consumed in microjoules
energy_consumed=$((final_energy - initial_energy))

# Convert energy consumed to microjoules
energy_consumed_in_microjoules=$energy_consumed

# Calculate total instructions executed (MIPS * duration in seconds * 1,000,000 for MIPS)
total_instructions=$(echo "$mips * $duration * 1000000" | bc)

# Calculate microjoules per instruction
microjoules_per_instruction=$(echo "scale=6; $energy_consumed_in_microjoules / $total_instructions" | bc)

# Output the results
echo "------------------- Performance Metrics -------------------"
echo "Energy Consumed: $energy_consumed_in_microjoules microjoules"
echo "MIPS: $mips"
echo "Duration: $duration sec"
echo "Total Instructions: $total_instructions"
echo "Microjoules per Instruction: $microjoules_per_instruction"
echo "-----------------------------------------------------------"
