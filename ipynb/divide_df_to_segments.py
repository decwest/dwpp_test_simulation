import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path("/home/decwest/decwest_workspace/dwpp_test_simulation/data/dwpp/lookahead/dynamic_window_pure_pursuit_log.csv")

def divide_df_to_segments(df: pd.DataFrame, t_threshold: float) -> list:
    """
    Divide the DataFrame into segments based on time gaps exceeding the threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a 'timestamp' column.
        t_threshold (float): Time gap threshold to identify segments.
        
    Returns:
        list: List of DataFrame segments.
    """
    t = df["sec"].to_numpy() + df["nsec"].to_numpy() * 1e-9
    
    # diffをとる
    t_diff = np.diff(t, prepend=t[0])
    print(t_diff)
    
    # diffが閾値以上のindexでcsvを分割する
    split_indices = np.where(t_diff >= t_threshold)[0]
    
    # split_indices から各区間を作成
    split_points = np.r_[0, split_indices, len(df)]
    segments = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        segment = df.iloc[start:end].copy()
        if not segment.empty:
            segments.append(segment)
    
    print(f"Found {len(segments)} segments (threshold={t_threshold})")
    for i, segment in enumerate(segments):
        span = segment["sec"].iloc[-1] - segment["sec"].iloc[0]
        print(f" segment {i}: rows={len(segment)}, span={span:.3f}s")
    
    return segments

df = pd.read_csv(csv_path)
segments = divide_df_to_segments(df, t_threshold=1.0)

# 各segmentをcsvとして保存
output_dir = csv_path.parent
for i, segment in enumerate(segments):
    segment_path = output_dir / f"segment_{i:02d}.csv"
    segment.to_csv(segment_path, index=False)
    print(f"Saved segment {i} to {segment_path}")
