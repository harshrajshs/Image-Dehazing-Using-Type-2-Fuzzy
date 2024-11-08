from PIL import Image
import numpy as np
import math

class ImageProcessor:
    def __init__(self, image_path, patch_size, beta):
        self.image = Image.open(image_path)
        self.image_np = np.array(self.image).astype('float64') / 255.0
        self.patch_size = patch_size
        self.beta = beta
        self.pad = int((self.patch_size-1)/2)
        self.min_channel = self._get_min_channel()
        self.patch_dict = self._get_patches()
        self.L_estimated = self._global_atmospheric_light_estimation()

    def _get_min_channel(self):
        """
        Extracts the minimum channel for each pixel in the RGB image.

        Returns:
        - np.ndarray: A 2D array containing the minimum channel values for each pixel.
        """
        R, G, B = self.image_np[:, :, 0], self.image_np[:, :, 1], self.image_np[:, :, 2]
        return np.minimum(np.minimum(R, G), B)

    def _get_patches(self):
        """
        Generates patches of the min_channel image with padding to avoid border issues.

        Returns:
        - dict: A dictionary mapping (i, j) coordinates to 3x3 patches centered at each (i, j).
        """
        padded_min_channel = np.pad(self.min_channel, pad_width=self.pad, mode='edge')
        rows, cols = self.min_channel.shape
        patch_dict = {}
        for i in range(rows):
            for j in range(cols):
                patch = padded_min_channel[i:i+3, j:j+3]
                patch_dict[(i, j)] = patch
        return patch_dict

    def _similarity_function(self, omega, H_h, sigma):
        """
        Calculates a similarity measure based on Gaussian function.

        Parameters:
        - omega (float): The pixel intensity.
        - H_h (float): The h-middle mean for the patch.
        - sigma (float): The standard deviation of h-middle means.

        Returns:
        - float: The similarity value.
        """
        epsilon = 1e-10
        return np.exp(-0.5 * ((omega - H_h) / (sigma + epsilon)) ** 2)

    def _h_middle_means(self, patch):
        """
        Calculates the h-middle means for a kxk patch as per the specified formula.

        Parameters:
        - patch (np.ndarray): A kxk numpy array representing the pixel values in a patch.

        Returns:
        - np.ndarray: An array containing the h-middle means for this patch.
        """
        k = patch.shape[0]
        flat_patch = patch.flatten().astype(np.int32)
        num_pixels = k * k
        h_values = int((num_pixels + 1) / 2)
        
        center_index = num_pixels // 2
        center_value = flat_patch[center_index]
        sum = center_value
        l = center_index
        r = center_index
        h_middle_means_list = []
        
        for i in range(h_values):
            avg = sum / (2 * i + 1)
            l -= 1
            r += 1
            if l >= 0 and r < num_pixels:
                sum += (flat_patch[l] + flat_patch[r])
            h_middle_means_list.append(avg)
        
        return h_middle_means_list

    def _calculate_h_by_k_matrix(self, patch):
        """
        Calculates an h-by-k matrix based on similarity measures.

        Parameters:
        - patch (np.ndarray): A kxk numpy array representing the pixel values in a patch.

        Returns:
        - np.ndarray: An h-by-k matrix of similarity measures.
        """
        h_middle_means_list = self._h_middle_means(patch)
        flat_patch = patch.flatten()
        
        h, k = len(h_middle_means_list), len(flat_patch)
        result_matrix = np.zeros((h, k))
        sigma = np.std(h_middle_means_list)
        
        for i in range(h):
            for j in range(k):
                result_matrix[i, j] = self._similarity_function(flat_patch[j], h_middle_means_list[i], sigma)
        
        return result_matrix

    def _get_weights_for_each_patch(self, patch):
        """
        Calculates average weights for each pixel in a patch.

        Parameters:
        - patch (np.ndarray): A kxk numpy array representing the pixel values in a patch.

        Returns:
        - np.ndarray: 1D array of average weights for each pixel in the patch.
        """
        mat = self._calculate_h_by_k_matrix(patch)
        return np.mean(mat, axis=0)

    def _depth_map_estimate_for_each_pixel(self, patch):
        """
        Estimates the depth map for a given pixel using weighted average of patch values.

        Parameters:
        - patch (np.ndarray): A kxk numpy array representing the pixel values in a patch.

        Returns:
        - float: The estimated depth value for the pixel.
        """
        flat_patch = patch.flatten()
        weights = self._get_weights_for_each_patch(patch)
        
        weighted_sum = np.dot(flat_patch, weights)
        weight_sum = np.sum(weights)
        
        if weight_sum == 0:
            return weighted_sum
        
        return weighted_sum / weight_sum

    def _scene_transmission_for_each_pixel(self, patch):
        """
        Calculates the scene transmission for a pixel using depth estimation.

        Parameters:
        - patch (np.ndarray): A kxk numpy array representing the pixel values in a patch.

        Returns:
        - float: The scene transmission value for the pixel.
        """
        depth = self._depth_map_estimate_for_each_pixel(patch)
        return np.exp(-self.beta * depth)

    def _global_atmospheric_light_estimation(self):
        """
        Estimates the global atmospheric light based on minimum values from h-by-k matrices.

        Returns:
        - float: The estimated global atmospheric light.
        """
        rows, cols = self.min_channel.shape
        L_total = 0
        for i in range(rows):
            for j in range(cols):
                mat = self._calculate_h_by_k_matrix(self.patch_dict[(i, j)])
                min_value = np.min(mat)
                L_total += min_value
        return L_total / (rows * cols)

    def enhance_image(self):
        """
        Enhances the image using scene transmission and atmospheric light estimation.

        Returns:
        - Image: The enhanced image.
        """
        rows, cols = self.min_channel.shape
        enhanced_image_np = np.zeros((rows, cols, 3))
        
        for i in range(rows):
            for j in range(cols):
                T = self._scene_transmission_for_each_pixel(self.patch_dict[(i, j)])
                for k in range(3):
                    enhanced_image_np[i][j][k] = ((self.image_np[i][j][k] - self.L_estimated) / T + self.L_estimated)
        
        # Scaling to 8-bit range
        enhanced_image_scaled = (enhanced_image_np - np.min(enhanced_image_np)) / \
                                (np.max(enhanced_image_np) - np.min(enhanced_image_np)) * 255
        enhanced_image_scaled = enhanced_image_scaled.astype(np.uint8)
        return Image.fromarray(enhanced_image_scaled)