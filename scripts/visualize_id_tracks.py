import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import cv2


from itertools import groupby
from operator import itemgetter

gt = pd.read_csv("data/video_valset_cropped/val_labels.csv")
unique_videos = gt['experiment'].unique().tolist()
path_to_videos = "/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/Alpha/"
path_to_output = "output/id_track_visualizations/"

# Define columns and colors
individual_columns = ['Flo', "Rab", "Gen", "Red", "Geo", "Her", "Cha", "Pin", "Mar", "Dar", "Nis", "Unm", "Uns", "Uns"]
colors = [
    "#889106",  # Flo
    "#8B0000",  # Rab (darkred)
    "#800080",  # Gen (purple)
    "#00008B",  # Red (darkblue)
    "#000000",  # Geo (black)
    "#C0C0C0",  # Her (silver)
    "#0000FF",  # Cha (blue)
    "#ADD8E6",  # Pin (lightblue)
    "#F08080",  # Mar (lightcoral / light red)
    "#008000",  # Dar (green)
    "#006400",  # Nis (darkgreen)
    "#808080",  # Unm (gray)
    "#808080",  # Uns (gray)
    "#808080"   # Uns (gray) duplicate for consistency
]

for vid in unique_videos:
    print(f"Processing video: {vid}")
    #label_path = os.path.join(path_to_labels, vid)
    #video_path = os.path.join(path_to_videos, vid.replace(".txt", ".mp4"))
    df = pd.read_csv("A_identification_results.csv")
    if 'experiment' not in df.columns:
        raise KeyError("DataFrame missing required column: 'experiment'")
    df = df[df['experiment'] == vid].reset_index(drop=True)

    if 'logits' in df.columns:
        df = df.drop(columns=['logits'])

    target_names = individual_columns[:12]

    probs_expanded = (
        df['probs']
        .astype(str)
        .str.replace(r'^\[+|\]+$', '', regex=True)  # remove leading/trailing brackets
        .str.strip()
        .str.split(r'\s+', expand=True)
        .iloc[:, :12]             # take at most first 12 entries
    )
    probs_expanded = probs_expanded.reindex(columns=range(12))
    probs_expanded.columns = target_names
    probs_expanded = probs_expanded.apply(pd.to_numeric)
    df = pd.concat([df.drop(columns=['probs']), probs_expanded], axis=1)

    df.rename(columns={'frame_num': 'frame'}, inplace=True)

    #df.columns = ["frame", "track_id", "V3", "V4", "V5", "V6", "conf", "class", "Cha", "Flo", "Gen", "Geo", "Her", "Rab", "Red", "Uns", "ID"]

    gt_vid = gt[gt['experiment'] == vid]


    gt_vid = gt_vid[['track_id', 'label']].drop_duplicates(subset='track_id').reset_index(drop=True)


    for row_number in range(len(gt_vid)):
        ind_id = gt_vid['track_id'].iloc[row_number]

        df_ind = df[df['track_id'] == ind_id]

        track_length = len(df_ind)
        print(track_length)
        
        #largest_bbox_size = round((df_ind['V5'] * df_ind['V6']).max())
        individual_name = gt_vid['label'].iloc[row_number]

        # Sort and sample 10 frames
        df_ind = df_ind.sort_values(by='frame')
        print("df_ind elements:", len(df_ind))
        print(df_ind.sample(10))
        # Get 10 equidistant positions (floats)
        num_samples = 10
        indices = np.linspace(1, len(df_ind) - 2, num=num_samples)

        # Round to nearest index and make unique
        rounded_indices = np.round(indices).astype(int)
        rounded_indices = np.clip(rounded_indices, 0, len(df_ind) - 1)
        print(rounded_indices)
        # Select rows at those indices
        df_sampled = df_ind.iloc[rounded_indices].drop_duplicates(subset='frame').sort_values(by='frame')
        print(df_sampled)
        # Open video
        #video = cv2.VideoCapture(video_path)

        # Create figure layout
        fig = plt.figure(figsize=(25, 10))
        #gs = gridspec.GridSpec(3, 10, height_ratios=[2, 1, 2])  # 3 rows: images, bars, area
        gs = gridspec.GridSpec(3, 11, height_ratios=[2, 1, 2], width_ratios=[1]*10 + [0.5], hspace=0.1)

        # Store x coords for vertical lines later
        frame_numbers = df_sampled['frame'].tolist()

        # Plot images and bar charts
        for i, row in enumerate(df_sampled.itertuples()):
            frame_number = row.frame
            track_id = row.track_id
            exp_name = vid.replace('.txt', '')
            img_filename = f"{exp_name}_{int(frame_number):06d}_{track_id}.png"
            print(img_filename)
            img_path = os.path.join("data/video_valset_cropped/images", img_filename)

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Failed to read image {img_path}")
                continue

            cropped_object = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Image
            ax_img = plt.subplot(gs[0, i])
            ax_img.imshow(cropped_object)
            ax_img.axis('off')
            ax_img.set_title(f'Frame {frame_number}')

            # Bar chart
            ax_bar = plt.subplot(gs[1, i])
            values = df_ind[df_ind['frame'] == frame_number][target_names].iloc[0]
            ax_bar.bar(target_names, values, color=colors)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])

        # Area chart
        ax_area = plt.subplot(gs[2, :10])
        # Reverse the individual columns and colors
        print(target_names)
        print(colors)
        rev_target_names = target_names#[::-1]
        rev_colors = colors[:12]#[::-1]
        print(rev_target_names)
        print(rev_colors)


        
        
        # Create area chart with reversed stacking
        y = [df_ind[col] for col in rev_target_names]
        x = df_ind['frame']

        ax_area.stackplot(x, y, labels=rev_target_names, colors=rev_colors)
        ax_area.axis('off')  # Optional: remove axes


        # Detect missing frame ranges
        all_frames = np.arange(df_ind['frame'].min(), df_ind['frame'].max() + 1)
        existing_frames = df_ind['frame'].values
        missing_frames = np.setdiff1d(all_frames, existing_frames)
        print(existing_frames)
        print(missing_frames)

        for k, g in groupby(enumerate(missing_frames), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            start = group[0]
            end = group[-1] + 1  # to cover the full width
            ax_area.axvspan(start, end, color='black', alpha=0.7)

        # Add red vertical lines for sampled frames
        for frame in frame_numbers:
            ax_area.axvline(x=frame, color='red', linestyle='-', linewidth=1)

        # Clean up the chart
        ax_area.set_xlabel('Frame')
        ax_area.set_yticks([])  # remove y-axis ticks
        ax_area.set_title("")   # remove title
        ax_area.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=8, prop={'size': 16})

        # Remove spines (box lines)
        for spine in ax_area.spines.values():
            spine.set_visible(False)

        # Extend chart closer to edges
        ax_area.margins(x=0)  # removes default margins on x-axis
        ax_area.set_xlim(df_ind['frame'].min(), df_ind['frame'].max())  # tighten to frame range


        # Optional: Connect bars to vertical lines with dotted lines
        # This is a visual approximation only (based on subplot bounding boxes)
        # Draw connection lines from each bar plot to the corresponding frame in the area chart
        for i, frame in enumerate(frame_numbers):
            # Get the bar axis position
            bar_ax = plt.subplot(gs[1, i])
            bar_bbox = bar_ax.get_position()
            bar_x_fig = (bar_bbox.x0 + bar_bbox.x1) / 2
            bar_y_fig = bar_bbox.y0  # bottom of bar chart

            # Compute x pixel position of frame in area chart
            area_x_data = frame
            area_y_data = 1  # top of the area chart

            # Convert data coordinates to figure coordinates
            area_x_fig, area_y_fig = ax_area.transData.transform((area_x_data, area_y_data))
            area_x_fig, area_y_fig = fig.transFigure.inverted().transform((area_x_fig, area_y_fig))

            # Now draw a line from the bar chart to the corresponding x-location in area chart
            fig.lines.append(plt.Line2D(
                [bar_x_fig, area_x_fig],
                [bar_y_fig, area_y_fig],
                transform=fig.transFigure,
                linestyle='dotted',
                color='black',
                linewidth=1
            ))

        valid_columns = target_names.copy()

        # Weighting function
        def calculate_weight(confidence):
            return np.exp(9.2 * (confidence - 0.5))

        def compute_weighted_summary(df):
            weighted_scores = dict.fromkeys(valid_columns, 0.0)  # initialize weights to 0

            for _, row in df.iterrows():
                row_vals = pd.to_numeric(row[valid_columns], errors='coerce')
                top_id = row_vals.idxmax()
                top_val = row_vals.max()

                weight = calculate_weight(top_val)
                weighted_scores[top_id] += weight

            return pd.Series(weighted_scores)

        summary = compute_weighted_summary(df_ind)
        ax_summary = plt.subplot(gs[2, 10])  # Last column in bottom row

        ax_summary.barh(summary.index, summary.values, color=colors[:len(summary)], height=0.6)
        ax_summary.invert_yaxis()  # Most important at top
        ax_summary.set_title("weighted \nSummary", fontsize=10)
        ax_summary.tick_params(labelsize=8)
        ax_summary.set_xlim(left=0)  # start bars from 0
        ax_summary.set_yticks([])  # remove y-axis ticks


        fig.suptitle(f"Individual Name (annotated): {individual_columns[individual_name]} \nTrack Length: {track_length}", #, Largest BBox Size: {largest_bbox_size},
                    fontsize=20, y=0.95)

        if individual_name in target_names:
            color_index = target_names.index(individual_name)
            individual_color = colors[color_index]
            
            # Add a colored square
            fig.patches.extend([
                Rectangle(
                    (0.6, 0.93), 0.015, 0.02,  # (x, y), width, height in figure coordinates
                    transform=fig.transFigure,
                    facecolor=individual_color,
                    linewidth=0.5
                )
            ])
            
        # Final layout
        #plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        # Ensure the output directory exists
        os.makedirs(path_to_output, exist_ok=True)

        # Save the figure
        #output_file = os.path.join(path_to_output, f"{vid.replace('.txt', '')}_{individual_name}_{ind_id}.jpg")
        output_file = os.path.join(path_to_output, f"{vid.replace('.txt', '')}_{individual_name}_{ind_id}.pdf")
        plt.savefig(output_file, bbox_inches='tight')

        # Close the figure to free memory
        plt.close(fig)

