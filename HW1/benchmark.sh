#!/bin/bash

# Check if a program was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <program> [arguments...]"
    exit 1
fi

# Extract the program name and its arguments
PROGRAM="$1"
shift  # Shift arguments so $@ now contains only the program arguments

# Check if the program exists and is executable
if [ ! -x "$PROGRAM" ]; then
    echo "Error: Program '$PROGRAM' not found or not executable."
    exit 1
fi

# Initialize accumulators
total_real_time=0
total_ratio=0

echo "Running $PROGRAM 10 times with arguments: $@"

for i in {1..10}; do
    # Run the program with arguments and capture timing info
    output=$( (time -p $PROGRAM "$@") 2>&1 )
    
    # Extract real, user, and system time from output
    real_time=$(echo "$output" | grep "real" | awk '{print $2}')
    user_time=$(echo "$output" | grep "user" | awk '{print $2}')
    sys_time=$(echo "$output" | grep "sys" | awk '{print $2}')

    # Ensure floating-point division for the ratio
    ratio=$(echo "scale=15; $sys_time / $user_time" | bc -l)

    # Accumulate times for averaging (ensure they are in float)
    total_real_time=$(echo "$total_real_time + $real_time" | bc)
    total_ratio=$(echo "$total_ratio + $ratio" | bc)

    # Print results for this run
    echo "Run #$i: Real Time = $real_time sec, System/User Ratio = $ratio"
done

# Calculate averages
avg_real_time=$(echo "scale=6; $total_real_time / 10" | bc)
avg_ratio=$(echo "scale=6; $total_ratio / 10" | bc)

# Output final results
echo "------------------------------------------------------"
echo "Average Runtime: $avg_real_time sec"
echo "Average System/User Time Ratio: $avg_ratio"
echo "------------------------------------------------------"

