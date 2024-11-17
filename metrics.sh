#!/bin/bash

# Function to extract and print metrics for a given section
print_metrics() {
    echo "$1" # Print the kernel name
    echo "$2" | awk '
        $1 ~ /^(Memory|DRAM|Duration|Compute)/ {
            printf "%-30s %10s %10s\n", $1, $2, $3
        }
    '
    echo ""
}

# Read the input and process it
awk '
    /^[a-zA-Z_]+\(.*\)/ {
        if (kernel != "") {
            print kernel
            print section
            print ""
        }
        kernel = $0
        section = ""
    }
    /Section: GPU Speed Of Light Throughput/,/^$/ {
        if ($1 ~ /^(Memory|DRAM|Duration|Compute)/) {
            section = section $0 "\n"
        }
    }
    END {
        if (kernel != "") {
            print kernel
            print section
        }
    }
' | while IFS= read -r line; do
    if [[ "$line" =~ ^[a-zA-Z_]+\(.*\) ]]; then
        kernel="$line"
        metrics=""
    elif [[ -n "$line" ]]; then
        metrics+="$line"$'\n'
    else
        if [[ -n "$kernel" && -n "$metrics" ]]; then
            print_metrics "$kernel" "$metrics"
        fi
        kernel=""
        metrics=""
    fi
done
