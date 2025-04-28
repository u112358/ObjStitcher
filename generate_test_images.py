import numpy as np
import cv2
import random
import os

# ---------------------------
# Parameter Settings
# ---------------------------
frame_height = 2500    # Number of rows per frame (image height)
frame_width = 2000     # Image width
M = 300               # Number of materials to generate
prob = 0.2            # Probability of length change
length_change_range = (-100, 100)  # Range of length variation

# Base number of rows for each material's sections
base_p = int(300 * 10.4)  # Base black rows
base_q = int(45 * 10.4)   # Base white rows
max_x = 10                # Maximum random offset

if base_q < max_x:
    raise ValueError(
        "base_q must be >= max_x to ensure non-negative white rows")

# ---------------------------
# Generate Materials with Advanced Features
# ---------------------------
materials = []
boundary_records = []
absolute_boundary_rows = []  # Track absolute boundary positions
last_changed = False
current_absolute_position = 0

# Create output directories
out_dir = './test_images'
boundary_dir = './boundary_data'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(boundary_dir, exist_ok=True)

for i in range(M):
    # Random offset selection
    x = random.randint(0, max_x)
    p = base_p + x
    q = base_q - x

    # Controlled length variation
    should_change = (random.random() < prob) and (not last_changed)
    length_change = 0

    if should_change:
        length_change = random.randint(*length_change_range)
        # Prefer changing p, modify q only if necessary
        new_p = p + length_change
        if new_p > 0:
            p = new_p
        else:
            q = q - length_change  # Compensate to maintain total length change
        last_changed = True
    else:
        last_changed = False

    # Create material sections
    black_section = np.zeros((p, frame_width), dtype=np.uint8)
    white_section = 255 * np.ones((q, frame_width), dtype=np.uint8)
    material = np.vstack([black_section, white_section])
    materials.append(material)

    # Record boundary information
    boundary_row = current_absolute_position + p - 1  # Absolute boundary position
    boundary_records.append({
        'material_id': i+1,
        'p': p,
        'q': q,
        'total_rows': p+q,
        'boundary_row': boundary_row,
        'length_changed': should_change,
        'length_change': length_change if should_change else 0
    })
    absolute_boundary_rows.append(boundary_row)
    current_absolute_position += p + q

    print(f"Material {i+1}: {p} black, {q} white, total {p+q} rows",
          f"(changed by {length_change})" if should_change else "")

# ---------------------------
# Create Continuous Roll and Save Boundary Data
# ---------------------------
roll = np.vstack(materials)
roll_height = roll.shape[0]
print(f"\nTotal lines in roll: {roll_height}")

# Save boundary data to file
with open(os.path.join(boundary_dir, 'boundary_data.csv'), 'w') as f:
    f.write(
        "material_id,absolute_boundary_row,p,q,total_rows,length_changed,length_change\n")
    for record in boundary_records:
        f.write(
            f"{record['material_id']},{record['boundary_row']},{record['p']},")
        f.write(
            f"{record['q']},{record['total_rows']},{int(record['length_changed'])},{record['length_change']}\n")

# ---------------------------
# Sliding Window Frame Capture with Boundary Visualization
# ---------------------------
start_index = random.randint(0, frame_height - 1)
frame_count = 0
i = start_index

while i + frame_height <= roll_height:
    frame = roll[i:i+frame_height, :]
    frame_count += 1

    # Visualize boundaries in frame (draw red line at boundary positions)
    for boundary in absolute_boundary_rows:
        if i <= boundary < i + frame_height:
            rel_pos = boundary - i
            cv2.line(frame, (0, rel_pos),
                     (frame_width-1, rel_pos), (0, 0, 255), 1)

    # Save frame
    filename = os.path.join(out_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(filename, frame)

    # Move to next frame (non-overlapping)
    i += frame_height

print(f"\nGenerated {frame_count} frames with boundary visualization")
print(
    f"Boundary data saved to: {os.path.join(boundary_dir, 'boundary_data.csv')}")
print(start_index)
