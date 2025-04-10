import torch

def compute_chromagram_from_cqt(cqt: torch.Tensor, bins_per_octave=12, num_octaves=7, avg_kernel_size=15):
    """
    Compute the chromagram from the CQT.

    Parameters
    ----------
    cqt : torch.Tensor
        The CQT of the audio signal. Shape: [batch_size, n_bins = 12 * 7, n_frames]
    bins_per_octave : int
        Number of bins per octave. Default is 12.
    num_octaves : int
        Number of octaves. Default is 7.
    avg_kernel_size : int
        Size of the moving average kernel. Default is 15.
    
    Returns
    -------
    torch.Tensor
        The relative chromagram. Shape: [batch_size, 12, n_frames]
    """
    assert cqt.shape[1] == bins_per_octave * num_octaves, f"cqt shape {cqt.shape} is not compatible with bins_per_octave {bins_per_octave} and num_octaves {num_octaves}"

    # Compute chromagram from cqt
    cqt = cqt.reshape(cqt.shape[0], num_octaves, bins_per_octave, cqt.shape[2])
    chromagram = cqt.mean(dim=1)  # batch, 12, n_frames

    # Moving average on chromagram
    avg_kernel = torch.ones(1, 1, avg_kernel_size, device=chromagram.device) / avg_kernel_size
    chromagram = torch.nn.functional.conv1d(chromagram, avg_kernel.expand(12, -1, -1), padding=avg_kernel_size // 2, groups=12)

    return chromagram


def shift_chromagram_by_root(chromagram: torch.Tensor, root: torch.Tensor):
    assert chromagram.shape[2] == root.shape[1], f"cqt {chromagram.shape} and root {root.shape} have different number of frames"

    # Shift the chromagram based on the root note
    root = root.unsqueeze(1).expand(-1, 12, -1)  # batch, 12, n_frames

    index = torch.arange(12, device=chromagram.device).view(1, 12, 1)
    shifted_index = (index + root) % 12
    chromagram = torch.gather(chromagram, dim=1, index=shifted_index)  # batch, 12, n_frames
    return chromagram

def compute_quality_pseudo_label(cqt: torch.Tensor, root: torch.Tensor, bins_per_octave=12, num_octaves=7, avg_kernel_size=15):
    """
    Compute the pseudo labels for the quality of the chord based on the CQT and root note.
    The algorithm simply looks at the relative energy of the maj 3rd and the min 3rd in the chromagram.

    Parameters
    ----------
    cqt : torch.Tensor
        The CQT of the audio signal. Shape: [batch_size, n_bins = 12 * 7, n_frames]
    root : torch.Tensor
        The root note of the chord. Shape: [batch_size, n_frames]
    bins_per_octave : int
        Number of bins per octave. Default is 12.
    num_octaves : int
        Number of octaves. Default is 7.
    avg_kernel_size : int
        Size of the moving average kernel. Default is 15.
    
    Returns
    -------
    torch.Tensor
        The pseudo labels for the quality of the chord. 0 for major, 1 for minor.
    """
    chromagram = compute_chromagram_from_cqt(cqt, bins_per_octave, num_octaves, avg_kernel_size)
    chromagram = shift_chromagram_by_root(chromagram, root)

    return (chromagram[:, 4, :] < chromagram[:, 3, :]).long()

def compute_root_pseudo_label(cqt: torch.Tensor, bins_per_octave=12, num_octaves=7, avg_kernel_size=15):
    """
    Compute the pseudo labels for the root note of the chord based on the CQT.
    The algorithm simply looks at the relative energy of the root note + fifth note in the chromagram.

    Parameters
    ----------
    cqt : torch.Tensor
        The CQT of the audio signal. Shape: [batch_size, n_bins = 12 * 7, n_frames]
    
    Returns
    -------
    torch.Tensor
        The pseudo labels for the root note of the chord. 0 for C, 1 for C#, ..., 11 for B.
    """
    chromagram = compute_chromagram_from_cqt(cqt, bins_per_octave, num_octaves, avg_kernel_size)

    fifth_index = (torch.arange(12, device=chromagram.device).view(1, 12, 1) + 7) % 12
    fifth_index = fifth_index.expand(chromagram.shape[0], -1, chromagram.shape[2])

    fifth_chromagram = torch.gather(chromagram, dim=1, index=fifth_index)  # batch, 12, n_frames
    root_chromagram = chromagram + fifth_chromagram
    
    return root_chromagram.argmax(dim=1).long()  # batch, n_frames