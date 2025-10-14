import torch
import numpy as np
from numpy import matlib
from typing import List, Tuple

# === Microphone array geometry (2D circular) ===
def circular_array_positions(radius: float, num_mics: int, reorder_idx_list: List[int]) -> torch.Tensor:
    """Return (num_mics, 3) xyz positions for a circular array in the xy-plane.

    Parameters
    ----------
    radius : float
        Array radius in meters.
    num_mics : int
        Number of microphones.
    reorder_idx_list : List[int]
        A permutation that maps the hardware channel order to the logical ring order.

    Returns
    -------
    torch.Tensor
        Positions as float32 tensor with z=0 for all mics.
    """
    angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
    mic_positions = torch.zeros((num_mics,3), dtype=torch.float32)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)
    for idx, reorder_idx in enumerate(reorder_idx_list):
        mic_positions[idx, :] = torch.FloatTensor([x[reorder_idx], y[reorder_idx], z[reorder_idx]])
    return mic_positions


def cartesian_to_azimuth_elevation(cartesian_coord: torch.Tensor) -> Tuple[float, float]:
    """Convert a 3D Cartesian vector to azimuth and elevation angles in degrees.

    Parameters
    ----------
    cartesian_coord : torch.Tensor
        A 3-element tensor [x, y, z].

    Returns
    -------
    Tuple[float, float]
        (azimuth_deg, elevation_deg)
    """
    cartesian_coord_list = cartesian_coord.detach().cpu().numpy()
    x,y,z = cartesian_coord_list
    # Compute azimuth in radians
    azimuth = np.arctan2(y, x)
    # Compute elevation in radians
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    # Convert radians to degrees
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)
    return azimuth_deg, elevation_deg

# return the direction of arrival [x,y,z] in meters
def compute_doa_from_location(speaker_str: str, mic_dist_str: str,
                            mic_name_str: str, num_windows: int,
                            z_drone: float = 1.5, z_src: float = 0.485,
                            ) -> torch.Tensor:
    """Compute per-window DOA vectors given geometry.

    Parameters
    ----------
    speaker_str : str
        String like "speaker-dist-1m"; determines x/y offsets.
    mic_dist_str : str
        String like "mic-dist-25cm"; determines mic z-offset relative to drone body.
    mic_name_str : str
        Contains 'down' or 'up' to select mic vertical offset direction.
    num_windows : int
        Number of STFT frames (time windows) to replicate the DOA vector across.
    z_drone : float, default=1.5
        Drone altitude in meters.
    z_src : float, default=0.485
        Source height in meters.

    Returns
    -------
    torch.Tensor
        DOA tensor of shape [1, num_windows, 3].
    """
    x_mic = 0; y_mic = 0
    z_drone_to_mic = int(mic_dist_str.split('-')[-1][:-2])/100. # convert to meters
    if 'down' in mic_name_str:
        z_mic = z_drone - z_drone_to_mic
    elif 'up' in mic_name_str:
        z_mic = z_drone + z_drone_to_mic
    else:
        print(f'Incorrect mic type: {mic_name_str}')
    x_src = -int(speaker_str.split('-')[-1][:-1])/1.414
    y_src = x_src
    doa = np.array([x_src-x_mic, y_src-y_mic, z_src-z_mic])
    azim, _ = cartesian_to_azimuth_elevation(torch.tensor(doa, dtype=torch.float32))
    # current data collection fixes the azimuth at 135 degrees
    assert np.abs(azim+135) < 1e-5, "Wrong azimuth value!"
    doas = matlib.repmat(doa, m=num_windows, n=1)
    doas = torch.tensor(doas, dtype=torch.float32)
    doas = doas.unsqueeze(0)
    return doas