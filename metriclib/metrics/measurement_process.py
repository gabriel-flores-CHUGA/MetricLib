import numpy as np
import torch
import antropy as ant
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from ..metric import MetricResult, StreamMetric, TabularMetric

import ast

class LimitofQuantification(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        if metric_config["cp"] is None:
            cp = 10
        else:
            cp = metric_config["cp"]

        if metric_config["LoB"] is None:
            raise ValueError("metric_config must include 'LoB' key")
        else:
            LoB = metric_config["LoB"]

        LoQ = LoB + cp * datapoint[0].mean()

        return LoQ

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster=None,
            threshold=0,
            description="Proportion of values below limit of quantification",
            value=np.array(data).mean(),
        )


class SampleEntropy(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        metric_config = metric_config or {}
        max_points = int(metric_config.get("max_points_per_lead", 500))

        values = []
        for i in range(datapoint[0].shape[0]):
            x = np.asarray(datapoint[0][i, :], dtype=np.float64)
            if max_points > 1 and x.size > max_points:
                step = int(np.ceil(x.size / max_points))
                x = x[::step]
            x = np.ascontiguousarray(x)
            m = int(2)
            r = float(0.2 * np.std(x))
            values.append(
                ant.sample_entropy(x, order=m, tolerance=r, metric="chebyshev")
            )

        entropy = float(np.mean(values))

        return entropy

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster="MeasurementProcess",
            threshold=0,
            description="Mean sample entropy across all leads",
            value=np.array(data).mean(),
        )


class SNR(StreamMetric):
    def aggregate(self, datapoint, reference=None, metric_config=None):
        if reference is None:
            raise ValueError("Reference signal is required for SNR calculation.")

        signal_power = np.mean(datapoint[0] ** 2)
        noise_power = np.mean((datapoint[0] - reference[0]) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def compute(self, data, reference, metric_config):
        return MetricResult(
            cluster=None,
            threshold=0.001,
            description="Mean SNR across all leads",
            value=np.array(data).mean(),
        )


class MetadataCompleteness(TabularMetric):
    def compute(self, data, reference=None, metric_config=None):
        # count missing values in relation to all cells in the DataFrame
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)

        return MetricResult(
            description="Metadata Completeness",
            value=completeness,
            cluster="Measurement Process",
            threshold=1.0,
        )

class ImageEntropy(StreamMetric):
    """
    Computes Entropy of a CT Scan image (NIFTI format).
    """
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - source image in a tensor (datapoint[0])
        Optional :
        - number of bins for entropy approximation (in metric_config["bins"])(default : 256)
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([array_img]) ; 
            reference : None.
            metric_config (example): {'bins' : 128}, 
        """
            
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if metric_config is None or metric_config["bins"] is None:
            bins = 256
        else :
            bins = int(metric_config["bins"])
        
        arr = (datapoint[0][0]).numpy()
        
        norm_arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        hist, _ = np.histogram(norm_arr.flatten(), bins=bins, density=False)
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-12)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    
    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Image entropy from a CT Scan image (NIFTI format)",
            value=data,
        )
        return res

class MeanGradientMagnitudeScale(StreamMetric):
    """
    Computes Mean Gradient Magnitude Scale of a CT Scan image (NIFTI format).
    We use 3 different scales to smooth the CT Scan, to help detect noise. 
    This metric takes into account noise level, image texture and image edges.
    Quantity of high-frequence content.
    
    Interpretation : 
        high value : a lot of details, maybe noise
        low value : smooth image
        
    By default : 
        sigma = 0.5 : little smoothing, get details 
        sigma = 1 : medium smoothing
        sigma = 2 : high smoothing, get large structures (ex : organs)
    """
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - source image in a tensor (datapoint[0])
        Optional :
        - tuples of sigmas values for smoothing (in metric_config["sigmas"])(default : (0.5, 1.0, 2.0))
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([array_img]) ; 
            reference : None.
            metric_config (example): {'sigmas' : (0.5, 1.0, 2.0)}, 
        """
            
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if metric_config is None or metric_config["sigmas"] is None:
            sigmas = (0.5, 1.0, 2.0)
        else :
            sigmas = int(metric_config["sigmas"])
        
        arr = (datapoint[0][0]).numpy()
        image_sitk = sitk.GetImageFromArray(arr)
        grads = []
        for s in sigmas:
            smoothed = sitk.DiscreteGaussian(image_sitk, s)
            grad = sitk.GradientMagnitude(smoothed)
            grads.append(float(np.mean(sitk.GetArrayFromImage((grad)))))
        return float(np.mean(grads))
        

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Mean Gradient Magnitude with Scales from a CT Scan image (NIFTI format)",
            value=data,
        )
        return res
    
class _TaskTransferFunction_tools(StreamMetric):
    """
    Toolkit for TaskTranferFunction classes
    """
    def compute_ttf_3d_fast(
        self,
        volume,
        max_profiles=1000,
        profile_half_length=10,
        sampling_step=3,
    ):
        Z, Y, X = volume.shape

        # --- 1. gradient
        gz, gy, gx = np.gradient(volume)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # --- 2. avoid flat zones
        threshold = np.percentile(grad_mag, 98)

        profiles = []

        # --- 3.  undersampled browsing
        for z in range(profile_half_length, Z - profile_half_length, sampling_step):
            for y in range(profile_half_length, Y - profile_half_length, sampling_step):
                for x in range(profile_half_length, X - profile_half_length, sampling_step):

                    if grad_mag[z, y, x] < threshold:
                        continue

                    direction = np.array([gz[z, y, x], gy[z, y, x], gx[z, y, x]])
                    norm = np.linalg.norm(direction)

                    if norm < 1e-6:
                        continue

                    direction = direction / norm

                    profile = []
                    valid = True

                    for t in range(-profile_half_length, profile_half_length + 1):
                        pos = np.array([z, y, x]) + t * direction
                        zi, yi, xi = np.round(pos).astype(int)

                        if (
                            0 <= zi < Z and
                            0 <= yi < Y and
                            0 <= xi < X
                        ):
                            profile.append(volume[zi, yi, xi])
                        else:
                            valid = False
                            break

                    if valid:
                        profiles.append(profile)

                    # stop early
                    if len(profiles) >= max_profiles:
                        break
                if len(profiles) >= max_profiles:
                    break
            if len(profiles) >= max_profiles:
                break

        if len(profiles) < 20:
            raise ValueError("Not enough profiles")

        profiles = np.array(profiles)

        # --- 4. alignment
        aligned = []
        for p in profiles:
            g = np.gradient(p)
            center = np.argmax(np.abs(g))
            shift = len(p)//2 - center
            aligned.append(np.roll(p, shift))

        aligned = np.array(aligned)

        # --- 5. ESF (Edge Spread Function) : how an ideal border is smoothed by the scanner
        esf = np.mean(aligned, axis=0)

        # light smoothing
        esf = np.convolve(esf, np.ones(3)/3, mode='same')

        # --- 6. LSF (Edge Spread Function)
        lsf = np.gradient(esf)

        # --- 7. FFT : transform into frequencies
        ttf = np.abs(np.fft.fft(lsf))
        ttf /= np.max(ttf)

        ttf = ttf[:len(ttf)//2]
        freqs = np.arange(len(ttf)) / len(ttf)

        return ttf, freqs

    def compute_ttf_metrics_interp(self, ttf, freqs, ttf_value):
        # interpolation

        ttf = np.array(ttf)
        freqs = np.array(freqs)

        def interp_x(target):
            # linear interpolation function

            # find where the curve cross the value
            for i in range(len(ttf) - 1):
                if (ttf[i] >= target and ttf[i+1] <= target):

                    # linear interpolation
                    x1, x2 = freqs[i], freqs[i+1]
                    y1, y2 = ttf[i], ttf[i+1]

                    return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

            return np.nan  # if not found

        ttf_computed_val = interp_x(ttf_value)
        return ttf_computed_val

class TaskTransferFunction50(_TaskTransferFunction_tools):
    """
    Computes Task Transfer Function 50 of a CT Scan image (NIFTI format).
    
    TTF50 and TTF10 are scalar metrics derived from the Task Transfer Function (TTF),
    which describes how well an imaging system preserves contrast at different
    spatial frequencies.

    - TTF50: Frequency at which the TTF falls to 50% of its maximum.
            Reflects effective spatial resolution (image sharpness).
            Low values : < 0.15
            High values : > 0.25

    Higher values indicate better resolution performance, while lower values
    suggest increased blurring or smoothing.

    These metrics should be interpreted alongside noise measurements (e.g., NPS)
    to assess overall image quality.
    """
    def compute_ttf_3d_fast(
        self,
        volume,
        max_profiles=1000,
        profile_half_length=10,
        sampling_step=3,
    ):
        return super().compute_ttf_3d_fast(volume,
            max_profiles,
            profile_half_length,
            sampling_step,
        )
    
    def compute_ttf_metrics_interp(self, ttf, freqs, ttf_value):
        return super().compute_ttf_metrics_interp(ttf, freqs, ttf_value)
    
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - base image in a tensor (datapoint[0])
        Optional :
        In a dictionary "metric_config" :
        - int max_profiles
        - int profile_half_length
        - int sampling_step
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([img]) ; 
            reference : None.
            metric_config (example): {'max_profiles' : 1000, 'profile_half_length':10, 'sampling_step':3}
        """
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        
        if metric_config is None or metric_config["max_profiles"] is None:
            max_profiles = 1000
        else :
            max_profiles = int(metric_config["max_profiles"])
            
        if metric_config is None or metric_config["profile_half_length"] is None:
            profile_half_length = 10
        else :
            profile_half_length = int(metric_config["profile_half_length"])
            
        if metric_config is None or metric_config["sampling_step"] is None:
            sampling_step = 3
        else :
            sampling_step = int(metric_config["sampling_step"])
            
        # arr = (datapoint[0][0]).numpy()
        arr = datapoint[0]
        
        ttf, freqs = self.compute_ttf_3d_fast(arr, max_profiles=max_profiles,
                                         profile_half_length=profile_half_length,
                                         sampling_step=sampling_step)
        ttf50 = self.compute_ttf_metrics_interp(ttf, freqs,0.5)
        return ttf50


    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Task Transfer Function 50 from a CT Scan image (NIFTI format)",
            value=data,
        )
        return res

class TaskTransferFunction10(_TaskTransferFunction_tools):
    """
    Computes Task Transfer Function 10 of a CT Scan image (NIFTI format).
    
    TTF50 and TTF10 are scalar metrics derived from the Task Transfer Function (TTF),
    which describes how well an imaging system preserves contrast at different
    spatial frequencies.

    - TTF10: Frequency at which the TTF falls to 10% of its maximum.
            Reflects limiting resolution (visibility of fine details).
            Low values : < 0.08
            High values : > 0.15

    Higher values indicate better resolution performance, while lower values
    suggest increased blurring or smoothing.

    These metrics should be interpreted alongside noise measurements (e.g., NPS)
    to assess overall image quality.
    """
    def compute_ttf_3d_fast(
        self,
        volume,
        max_profiles=1000,
        profile_half_length=10,
        sampling_step=3,
    ):
        return super().compute_ttf_3d_fast(volume,
            max_profiles,
            profile_half_length,
            sampling_step,
        )
    
    def compute_ttf_metrics_interp(self, ttf, freqs, ttf_value):
        return super().compute_ttf_metrics_interp(ttf, freqs, ttf_value)
    
    
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - base image in a tensor (datapoint[1])
        Optional :
        In a dictionary "metric_config" :
        - int max_profiles
        - int profile_half_length
        - int sampling_step
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([img]) ; 
            reference : None.
            metric_config (example): {'max_profiles' : 1000, 'profile_half_length':10, 'sampling_step':3}
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        
        if metric_config is None or metric_config["max_profiles"] is None:
            max_profiles = 1000
        else :
            max_profiles = int(metric_config["max_profiles"])
            
        if metric_config is None or metric_config["profile_half_length"] is None:
            profile_half_length = 10
        else :
            profile_half_length = int(metric_config["profile_half_length"])
            
        if metric_config is None or metric_config["sampling_step"] is None:
            sampling_step = 3
        else :
            sampling_step = int(metric_config["sampling_step"])
            
        
        arr = datapoint[0]
        
        # arr = (datapoint[0][0]).numpy()
        
        
        ttf, freqs = self.compute_ttf_3d_fast(arr, max_profiles=max_profiles,
                                         profile_half_length=profile_half_length,
                                         sampling_step=sampling_step)
        ttf10 = self.compute_ttf_metrics_interp(ttf, freqs,0.1)
        return ttf10


    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Task Transfer Function 10 from a CT SCan image (NIFTI format)",
            value=data,
        )
        return res
    
class NoisePowerSpectrum(StreamMetric):
    """
    Computes Noise Power Spectrum of a CT Scan image (NIFTI format).
    
    The NPS describes how noise is distributed across spatial frequencies,
    providing both the magnitude and texture of noise.

    Key outputs:
        - NPS(f): frequency-dependent noise power
        - total_power: overall noise level (proportional to noise variance)

    Interpretation:
        - Higher total_power → more noise
        - Lower total_power → cleaner image
        - Low-frequency NPS → smooth noise
        - High-frequency NPS → fine, grainy noise

    The NPS is a standard metric in CT image quality assessment and should
    be interpreted together with resolution metrics (e.g., TTF).
    
    By default : 
        patch_size = 32
        stride = 16
        keep_fraction = 0.1
        apply_window = True
        
    """

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - base image in a tensor (datapoint[0])
        Optional :
        In a dictionary "metric_config" :
        - int patch_size
        - int stride
        - float keep_fraction
        - bool apply_window
        
        Raises :
            ValueError : if "datapoint[0]" is not a torch.tensor
        Args:
            datapoint : 
                datapoint[0] : torch.tensor([array_img]) ; 
            reference : None.
            metric_config (example): {'patch_size' : 32, 'stride':16, 'keep_fraction':0.1, 'apply_window':True }
            
        """
        if not torch.is_tensor(datapoint[0]):
            raise ValueError("Data must be structured in a torch.tensor.")
        
        if metric_config is None or metric_config["patch_size"] is None:
            patch_size = 32
        else :
            patch_size = int(metric_config["patch_size"])
            
        if metric_config is None or metric_config["stride"] is None:
            stride = 16
        else :
            stride = int(metric_config["stride"])
            
        if metric_config is None or metric_config["keep_fraction"] is None:
            keep_fraction = 0.1
        else :
            keep_fraction = float(metric_config["keep_fraction"])
            
        if metric_config is None or metric_config["apply_window"] is None:
            apply_window = True
        else :
            apply_window = bool(metric_config["apply_window"])
            
        
        arr = datapoint[0]
        # arr = (datapoint[0][0]).numpy()
        arr = np.asarray(arr, dtype=np.float32)
        
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]

        H, W = arr.shape

        patches = []
        scores = []

        # --- Hanning Window
        if apply_window:
            win = np.hanning(patch_size)
            window = np.outer(win, win)
        else:
            window = np.ones((patch_size, patch_size))

        # --- 1. Extract patches
        for i in range(0, H - patch_size, stride):
            for j in range(0, W - patch_size, stride):

                patch = arr[i:i+patch_size, j:j+patch_size]

                gx, gy = np.gradient(patch)
                grad = np.sqrt(gx**2 + gy**2)

                score = np.mean(grad) + 0.5 * np.var(patch)

                patches.append(patch)
                scores.append(score)

        patches = np.array(patches)
        scores = np.array(scores)

        if len(patches) == 0:
            raise ValueError("No patch extracted.")
        

        # --- 2. Select best patches
        N_keep = max(10, int(len(patches) * keep_fraction))
        idx = np.argsort(scores)[:N_keep]
        selected_patches = patches[idx]
    
        # --- 3. Compute NPS
        nps_list = []

        for patch in selected_patches:
            patch = patch - np.mean(patch)
            patch = patch * window

            fft = np.fft.fftshift(np.fft.fft2(patch))
            nps = (np.abs(fft) ** 2) / (patch_size ** 2)

            nps_list.append(nps)

        nps_2d_mean = np.mean(nps_list, axis=0)

        # --- 4. Radial averaging
        y, x = np.indices(nps_2d_mean.shape)
        center = np.array(nps_2d_mean.shape) // 2
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(np.int32)

        tbin = np.bincount(r.ravel(), nps_2d_mean.ravel())
        nr = np.bincount(r.ravel())

        nps_1d = tbin / np.maximum(nr, 1)
        freqs = np.arange(len(nps_1d)) / patch_size

        # --- Metrics
        total_power = float(np.sum(nps_1d))

        # Mean frequency (center of mass of spectrum)
        mean_freq = float(np.sum(freqs * nps_1d) / np.sum(nps_1d))

        # Peak frequency
        peak_freq = float(freqs[np.argmax(nps_1d)])

        # --- Classification (simple rule)
        # empirical value
        if mean_freq < 0.15:
            noise_type = "low_frequency"   # smoothed noise / IR
        else:
            noise_type = "high_frequency"  # grainy noise / FBP

        metrics = {
            "num_total_patches": len(patches),
            "num_selected_patches": len(selected_patches),
            "total_power": total_power,
            "mean_frequency": mean_freq,
            "peak_frequency": peak_freq,
            "noise_type": noise_type,
            "mean_score_selected": float(np.mean(scores[idx])),
        }

        return nps_1d, freqs, metrics


    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Noise Power Spectrum from a CT Scan image (NIFTI format)",
            value=data,
        )
        return res
    

class DICESimilarityCoefficient(StreamMetric):
    """
    Computes de DSC between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction

        overlap.Execute(seg1_img, seg2_img)
        dice = overlap.GetDiceCoefficient()
        return dice

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster="Measurement Process",
            threshold=0,
            description="DICE Score between two segmentations",
            value=data,
        )
        return res

class IntersectionOverUnion(StreamMetric):
    """
    Computes de Intersection over Union score between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction

        overlap.Execute(seg1_img, seg2_img)
        iou = overlap.GetJaccardCoefficient()
        return iou

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Intersection over Union Score between two segmentations",
            value=data,
        )
        return res

class _HausdorffDistance_tools(StreamMetric):
    """
    Toolkit for HausdorffDistance classes
    """
    def _mask_to_surface_indices(self, mask_np):
        """
        Gives only surface mask from complete mask.
        Args:
            mask_np (np.Array): input mask, binary (z,y,x) array

        Returns:
            np.Array : indices in array index order (z,y,x)
        """
        eroded = binary_erosion(mask_np)
        surface = mask_np & (~eroded)
        inds = np.argwhere(surface)
        return inds

    def _indices_to_physical_points(self, img, inds):
        """
        Gives physical points (in mm) from indices in 3D sitk image
        Args:
            img (sitk Image): Segmentation image
            inds (np.Array): array Nx3 of (z,y,x), indices of surface voxels

        Returns:
            np.Array : shape (N,3) in mm (physical)
        """

        pts = [
            img.TransformIndexToPhysicalPoint((int(i[2]), int(i[1]), int(i[0])))
            for i in inds
        ]
        return np.array(pts)


    def _get_distances(self, seg1, seg2):

        # Distance maps (IMPORTANT: useImageSpacing=True → mm)
        seg1_dist = sitk.SignedMaurerDistanceMap(
            seg1, squaredDistance=False, useImageSpacing=True
        )
        seg2_dist = sitk.SignedMaurerDistanceMap(
            seg2, squaredDistance=False, useImageSpacing=True
        )

        # Surfaces (beaucoup plus robuste que ton extraction maison)
        seg1_surface = sitk.LabelContour(seg1)
        seg2_surface = sitk.LabelContour(seg2)

        # Conversion numpy
        seg1_surface_np = sitk.GetArrayViewFromImage(seg1_surface)
        seg2_surface_np = sitk.GetArrayViewFromImage(seg2_surface)

        seg1_dist_np = sitk.GetArrayViewFromImage(seg1_dist)
        seg2_dist_np = sitk.GetArrayViewFromImage(seg2_dist)

        # Distances surface → autre masque
        dists_seg1_to_seg2 = np.abs(seg2_dist_np[seg1_surface_np > 0])
        dists_seg2_to_seg1 = np.abs(seg1_dist_np[seg2_surface_np > 0])

        return dists_seg2_to_seg1, dists_seg1_to_seg2


class HausdorffDistance(_HausdorffDistance_tools):
    """
    Computes Hausdorff Distance (in mm) between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def _mask_to_surface_indices(self, mask_np):
        """
        Gives only surface mask from complete mask.
        Args:
            mask_np (np.Array): input mask, binary (z,y,x) array

        Returns:
            np.Array : indices in array index order (z,y,x)
        """
        return super()._mask_to_surface_indices(mask_np)

    def _indices_to_physical_points(self, img, inds):
        """
        Gives physical points (in mm) from indices in 3D sitk image
        Args:
            img (sitk Image): Segmentation image
            inds (np.Array): array Nx3 of (z,y,x), indices of surface voxels

        Returns:
            np.Array : shape (N,3) in mm (physical)
        """
        return super()._indices_to_physical_points(img, inds)

    def _get_distances(self, seg1, seg2):
        """
        Gives distances between two segmentations, in mm

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            np.Array,np.Array : gives distances from seg1->seg2 and from seg2->seg1
        """
        

        return super()._get_distances(seg1, seg2)

    def _getHD(self, seg1, seg2):
        """ 
        Compute Hausdorff Distance (in mm) between two segmentations.

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            float: Hausdorff Distance
        """
        dists_seg2_to_seg1, dists_seg1_to_seg2 = self._get_distances(seg1, seg2)
        hd_max = max(dists_seg2_to_seg1.max(), dists_seg1_to_seg2.max())

        return hd_max

    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction
        
        
        hd = self._getHD(seg1_img, seg2_img)
        return hd

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="maximum Hausdorff Distance between two segmentations",
            value=data,
        )
        return res
    
class HausdorffDistance95(_HausdorffDistance_tools):
    """
    Computes Hausdorff Distance 95 (in mm) between two segmentations.
    Needs to have two segmentation files in NIFTI format.
    In the dataset : segmentations are loaded with sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)).
    """

    def _mask_to_surface_indices(self, mask_np):
        """
        Gives only surface mask from complete mask.
        Args:
            mask_np (np.Array): input mask, binary (z,y,x) array

        Returns:
            np.Array : indices in array index order (z,y,x)
        """
        return super()._mask_to_surface_indices(mask_np)

    def _indices_to_physical_points(self, img, inds):
        """
        Gives physical points (in mm) from indices in 3D sitk image
        Args:
            img (sitk Image): Segmentation image
            inds (np.Array): array Nx3 of (z,y,x), indices of surface voxels

        Returns:
            np.Array : shape (N,3) in mm (physical)
        """
        return super()._indices_to_physical_points(img, inds)

    def _get_distances(self, seg1, seg2):
        """
        Gives distances between two segmentations, in mm

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            np.Array,np.Array : gives distances from seg1->seg2 and from seg2->seg1
        """
        

        return super()._get_distances(seg1, seg2)

    def _getHD95(self, seg1, seg2):
        """
        Compute Hausdorff Distance 95 (in mm) between two segmentations.
        HD95 removes 5% of extreme values compared to full Hausdorff Distance.

        Args:
            seg1 (sitk Image): first segmentation
            seg2 (sitk Image): second segmentation

        Returns:
            float: Hausdorff Distance 95
        """

        dists_seg2_to_seg1, dists_seg1_to_seg2 = self._get_distances(seg1, seg2)

        # Hausdorff symmetric
        hd95 = max(np.percentile(dists_seg2_to_seg1, 95), np.percentile(dists_seg1_to_seg2, 95))
        
        return hd95
            
    def aggregate(self, datapoint, reference=None, metric_config=None):
        """
        Requieres : 
        - segmentations images in a tensor (datapoint[1])
        - metric_config to have ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] keys.
        This keys are used to map the requiered information about segmentations to the right columns in the metadata file. 
        Therefore, we need specific information about images in the metadata file. It is accessible with simpleITK : 
            img_seg1 = sitk.ReadImage(seg1_path)
            seg1_origin = img_seg1.GetOrigin()
            seg1_spacing = img_seg1.GetSpacing()
            seg1_direction = img_seg1.GetDirection()
        
        Raises :
            ValueError : if "datapoint[1]" is not a torch.tensor
            ValueError : if len(datapoint[1]) != 2 : datapoint[1] must contains seg1 and seg2
            ValueError : if ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'] not in metric_config.keys() : metric_config must contains those keys
        Args:
            datapoint : 
                datapoint[1] : torch.tensor([seg1,seg2]) ; 
            reference : None.
            metric_config (example): {'seg1_origin' : 'col1', 'seg1_spacing':'col2', 'seg1_direction':'col3','seg2_origin' : 'col4', 'seg2_spacing':'col5', 'seg2_direction':'col6'}.
        """
        if not torch.is_tensor(datapoint[1]):
            raise ValueError("Data must be structured in a torch.tensor.")
        if len(datapoint[1]) != 2 :
            raise ValueError("Tensor must contains : segmentation 1 (array), segmentation 2 (array)")
        if metric_config["seg1_origin"] is None or metric_config["seg1_spacing"] is None or metric_config["seg1_direction"] is None or \
           metric_config["seg2_origin"] is None or metric_config["seg2_spacing"] is None or metric_config["seg2_direction"] is None :
            raise ValueError("metric_config must include the following keys : ['seg1_origin','seg1_spacing','seg1_direction','seg2_origin','seg2_spacing','seg2_direction'].")
        
        seg1_img = sitk.GetImageFromArray(datapoint[1][0])
        # rebuild segmentation specifications
        seg1_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg1_origin"])])) # origin
        seg1_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg1_spacing"])])) # spacing
        seg1_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg1_direction"])])) # direction

        seg2_img = sitk.GetImageFromArray(datapoint[1][1])
        # rebuild segmentation specifications
        seg2_img.SetOrigin(ast.literal_eval(datapoint[2][str(metric_config["seg2_origin"])])) # origin
        seg2_img.SetSpacing(ast.literal_eval(datapoint[2][str(metric_config["seg2_spacing"])])) # spacing
        seg2_img.SetDirection(ast.literal_eval(datapoint[2][str(metric_config["seg2_direction"])])) # direction
        
        
        hd95 = self._getHD95(seg1_img, seg2_img)
        return hd95

    def compute(self, data, reference, metric_config):
        res = MetricResult(
            cluster=None,
            threshold=0,
            description="Hausdorff Distance 95% \between two segmentations",
            value=data,
        )
        return res
