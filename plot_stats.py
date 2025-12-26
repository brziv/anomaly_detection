import json
import matplotlib.pyplot as plt
import os

# Path to the features.jsonl file
features_file = 'features.jsonl'

# Data structures
unique_videos = {}  # video_path -> {'duration': float, 'fps': float}
clip_durations = {'accident': [], 'normal': []}

# Read the JSONL file
with open(features_file, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        video_path = data['video_path']
        label = data['label']
        timestamp = data['timestamp']
        fps = data['fps']
        duration = data['duration']
        
        # For unique videos
        if video_path not in unique_videos:
            unique_videos[video_path] = {'duration': duration, 'fps': fps}
        
        # For clip durations
        clip_duration = timestamp[1] - timestamp[0]
        clip_durations[label].append(clip_duration)

# Plot 1: Histogram of video durations (unique videos)
durations = [info['duration'] for info in unique_videos.values()]
plt.figure(figsize=(8, 5))
plt.hist(durations, bins=30, edgecolor='black')
plt.title('Histogram of Video Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.yscale('log')
plt.savefig('video_duration_histogram.png')
plt.close()

# Plot 2: Grouped bar chart of video fps (unique videos)
fps_counts = {}
for info in unique_videos.values():
    fps = info['fps']
    fps_counts[fps] = fps_counts.get(fps, 0) + 1

fps_values = sorted([fps for fps in fps_counts if fps_counts[fps] > 0 and fps.is_integer()])
counts = [fps_counts[fps] for fps in fps_values]

plt.figure(figsize=(8, 5))
positions = range(len(fps_values))
plt.bar(positions, counts, edgecolor='black')
plt.title('Bar Chart of Video FPS')
plt.xlabel('FPS')
plt.ylabel('Count')
plt.yscale('log')
plt.xticks(positions, [str(int(fps)) for fps in fps_values])
plt.savefig('video_fps_bar_chart.png')
plt.close()

# Plot 3: Histogram of clip durations, colored by label
plt.figure(figsize=(10, 6))
plt.hist(clip_durations['normal'], bins=30, alpha=0.7, label='Normal', color='skyblue', edgecolor='black')
plt.hist(clip_durations['accident'], bins=30, alpha=0.7, label='Accident', color='salmon', edgecolor='black')
plt.title('Histogram of Clip Durations')
plt.xlabel('Clip Duration (seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('clip_duration_histogram.png')
plt.close()

print("Plots saved: video_duration_histogram.png, video_fps_bar_chart.png, clip_duration_histogram.png")