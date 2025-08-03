#!/bin/bash

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <video_list.txt>"
    echo "Please provide a txt file containing video IDs as argument"
    exit 1
fi

# Check if file exists and is a txt file
if [[ ! -f "$1" || "${1##*.}" != "txt" ]]; then
    echo "Error: Please provide a valid txt file"
    exit 1
fi

mkdir -p logs
mkdir -p data/ZeroMatch/pseudo
mkdir -p data/ZeroMatch/video_1080p

# Create GPU lock file directory
GPU_LOCK_DIR="/tmp/gpu_locks"
mkdir -p "$GPU_LOCK_DIR"

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

function cleanup_locks() {
    rm -f "$GPU_LOCK_DIR"/*.lock
    exit
}

# Set signal handler to cleanup lock files when script exits
trap cleanup_locks EXIT INT TERM

function select_gpu_safe() {
    local max_attempts=60  # Maximum 30 minutes of attempts
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        # Check memory status of all GPUs
        readarray -t total_memory < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        readarray -t memory_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        
        for i in "${!memory_free[@]}"; do
            local lock_file="$GPU_LOCK_DIR/gpu_${i}.lock"
            
            # Use file lock to avoid race conditions
            if (exec 200>"$lock_file"; flock -n 200); then
                local free_percent=$(awk -v free="${memory_free[$i]}" -v total="${total_memory[$i]}" 'BEGIN{print (free/total)*100}')
                
                # Lower threshold to 80%, more practical
                if (( $(awk -v fp="$free_percent" -v tp=80 'BEGIN{print (fp >= tp)}') )); then
                    # Create lock file and write process info
                    echo "$$:$(date)" > "$lock_file"
                    echo $i
                    return 0
                fi
            fi
        done
        
        attempt=$((attempt + 1))
        echo "All GPUs are busy, waiting 30 seconds... (attempt $attempt/$max_attempts)" >&2
        sleep 30
    done
    
    echo "Error: GPU wait timeout" >&2
    return 1
}

function release_gpu() {
    local gpu_id=$1
    local lock_file="$GPU_LOCK_DIR/gpu_${gpu_id}.lock"
    rm -f "$lock_file"
}

# Cleanup function after task completion
function run_task_with_cleanup() {
    local gpu=$1
    local cmd="$2"
    
    # Run task
    eval "$cmd"
    local exit_code=$?
    
    # Release GPU lock
    release_gpu "$gpu"
    
    return $exit_code
}

while IFS= read -r VIDEO_ID
do
    VIDEO_ID=$(echo "$VIDEO_ID" | tr -d '[:space:]')
    
    output_file="./data/ZeroMatch/video_1080p/${VIDEO_ID}.mp4"
    if [ ! -f "$output_file" ]; then
        yt-dlp -f 'bv*[ext=mp4][height=1080]+ba[ext=m4a]/b[ext=mp4][height=1080]' -S "height, fps" "https://www.youtube.com/watch?v=$VIDEO_ID" -o "$output_file"
    else
        echo "Video file $output_file already exists. Skipping download."
    fi
    
    printf "|%s|%s|%s|%s|%s|%s|\n" "======================" "==============" "============" "======" "========" "====="
    printf "| %-20s | %-12s | %-10s | %-4s | %-6s | %-3s |\n" "Timestamp" "Video ID" "Method" "Skip" "Resize" "GPU"
    printf "|%s|%s|%s|%s|%s|%s|\n" "----------------------" "--------------" "------------" "------" "--------" "-----"
    
    # Collect all tasks into array
    declare -a tasks=()
    
    # First round: no resize
    for skip in 0 1 2; do
        for method in GIM_DKM GIM_LOFTR GIM_GLUE SIFT; do
            tasks+=("$VIDEO_ID|$method|$skip|No")
        done
    done
    
    # Second round: with resize
    for skip in 0 1 2; do
        for method in GIM_DKM GIM_LOFTR GIM_GLUE SIFT; do
            tasks+=("$VIDEO_ID|$method|$skip|Yes")
        done
    done
    
    # Execute tasks in order
    for task in "${tasks[@]}"; do
        IFS='|' read -r vid method skip resize <<< "$task"
        
        # Safely acquire GPU
        gpu=$(select_gpu_safe)
        if [ $? -ne 0 ]; then
            echo "Error: Unable to acquire available GPU, skipping task $task" >&2
            continue
        fi
        
        logstamp=$(date +'%Y%m%d_%H%M%S')
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        printf "| %-20s | %-12s | %-10s | %-4s | %-6s | %-3s |\n" "$timestamp" "$vid" "$method" "$skip" "$resize" "$gpu"
        
        if [ "$resize" = "Yes" ]; then
            cmd="python video_preprocessor.py --gpu=$gpu --scene_name=\"$vid\" --method=$method --skip=$skip --resize > \"logs/${vid}_${method}_skip${skip}_resize_${logstamp}.log\" 2>&1"
        else
            cmd="python video_preprocessor.py --gpu=$gpu --scene_name=\"$vid\" --method=$method --skip=$skip > \"logs/${vid}_${method}_skip${skip}_${logstamp}.log\" 2>&1"
        fi
        
        # Run task in background, automatically release GPU lock after completion
        (run_task_with_cleanup "$gpu" "$cmd") &
        
        # Increase wait time to ensure task fully starts and occupies GPU memory
        sleep 60
    done
    
done < "$1"

# Wait for all background tasks to complete
wait

echo "All tasks completed!"