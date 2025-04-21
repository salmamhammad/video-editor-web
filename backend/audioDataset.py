import os
import torchaudio


# Custom dataset for noisy and clean audio pairs
class AudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
    
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.file_names = self._get_file_pairs()  

    def _get_file_pairs(self):
        noisy_files = set(os.listdir(self.noisy_dir))
        clean_files = set(os.listdir(self.clean_dir))
        common_files = noisy_files.intersection(clean_files)
        if not common_files:
            raise ValueError("No matching files found between noisy and clean directories.")
        
        return list(common_files)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
       
        file_name = self.file_names[idx]
        noisy_path = os.path.join(self.noisy_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_audio, _ = torchaudio.load(noisy_path)
        clean_audio, _ = torchaudio.load(clean_path)
        if self.transform:
            noisy_audio = self.transform(noisy_audio)
            clean_audio = self.transform(clean_audio)

        return noisy_audio, clean_audio

