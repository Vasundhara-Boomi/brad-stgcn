import os
import torch
from torch.utils.data import Dataset

class SingleVideoDataset(Dataset):
    def __init__(self, base_dir):
        """
        Initialize dataset for a single video.
        
        Args:
        - base_dir: Directory containing the video's tensor files
        """
        self.keypoints = []
        self.adjacency = []
        
        # Look for keypoints and adjacency files
        file_names = os.listdir(base_dir)
        
        # Find keypoints and adjacency files
        keypoints_file = next((f for f in file_names if "keypoints_" in f), None)
        adjacency_file = next((f for f in file_names if "adjacency_" in f), None)
        
        if not keypoints_file or not adjacency_file:
            raise RuntimeError("Could not find keypoints or adjacency tensor files")
        
        # Load tensors
        keypoints_path = os.path.join(base_dir, keypoints_file)
        adjacency_path = os.path.join(base_dir, adjacency_file)
        
        try:
            keypoints_tensor = torch.load(keypoints_path)
            adjacency_tensor = torch.load(adjacency_path)
            
            # Ensure tensors are in the right shape
            if keypoints_tensor.dim() == 3:
                keypoints_tensor = keypoints_tensor.unsqueeze(0)
            if adjacency_tensor.dim() == 2:
                adjacency_tensor = adjacency_tensor.unsqueeze(0)
            
            self.keypoints = keypoints_tensor
            self.adjacency = adjacency_tensor
            
            print(f"Loaded single video. Keypoints shape: {self.keypoints.shape}, Adjacency shape: {self.adjacency.shape}")
        
        except Exception as e:
            raise RuntimeError(f"Error loading tensors: {e}")
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        """
        Return keypoints and adjacency for the video.
        
        Args:
        - idx: Index (should always be 0)
        
        Returns:
        - Keypoints tensor
        - Adjacency tensor
        """
        if idx != 0:
            raise IndexError("This dataset contains only one video")
        
        return self.keypoints[0], self.adjacency[0]