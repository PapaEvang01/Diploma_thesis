# model_loader.py
# ----------------
# Utility class for loading the best CRAT-Pred model checkpoint (based on minimum FDE).
# It selects the checkpoint file with the lowest Final Displacement Error (fde_val),
# loads it, and returns the model in evaluation mode.

import os
import sys

# Append CRAT-Pred model directory to system path (adjust as needed for your setup)
sys.path.append("/home/user/PycharmProjects/crat-pred/model")

from crat_pred import CratPred  # Import the CRAT-Pred model architecture


class ModelLoader():
    def __init__(self):
        self.model = None  # Placeholder for the loaded model

    def load_cratpred_model(self):
        """
        Loads the CRAT-Pred model using the best available checkpoint.

        Returns:
            CratPred: The trained CRAT-Pred model in evaluation mode.
        """
        best_checkpoint = self.get_best_checkpoint()
        print(f"Loading best checkpoint: {best_checkpoint}")

        # Load model from checkpoint (non-strict to avoid mismatch warnings)
        model = CratPred.load_from_checkpoint(best_checkpoint, strict=False)
        model.eval()  # Set to evaluation mode (disables dropout, etc.)

        return model

    def get_best_checkpoint(self,
                            checkpoint_dir="/home/user/PycharmProjects/crat-pred/lightning_logs/version_2/checkpoints"):
        """
        Finds the best CRAT-Pred model checkpoint by selecting the one with the lowest fde_val.

        Args:
            checkpoint_dir (str): Path to the directory containing model checkpoint files.

        Returns:
            str: The path to the best checkpoint file.
        """
        # List all .ckpt files in the directory
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

        def extract_fde(filename):
            try:
                # Extract the float value after "fde_val=" from filename
                return float(filename.split("fde_val=")[-1].split("-")[0])
            except:
                return float("inf")  # If parsing fails, treat it as worst score

        # Select the checkpoint with the lowest FDE value
        best_checkpoint = min(checkpoints, key=extract_fde)
        return os.path.join(checkpoint_dir, best_checkpoint)

