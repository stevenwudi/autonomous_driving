# coding=utf-8
"""
Di Wu's re-implemenation of the Fast Discriminative Scale Space Tracker (fDSST) [1],
which is an extension of the VOT2014 winning DSST tracker [2].
The code provided by [3] is used for computing the HOG features.


[1]	Martin Danelljan, Gustav Fahad Khan, Michael Felsberg.
	Discriminative Scale Space Tracking.
	Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

[2] Martin Danelljan, Gustav Fahad Shahbaz Khan and Michael Felsberg.
    "Accurate Scale Estimation for Robust Visual Tracking".
    Proceedings of the British Machine Vision Conference (BMVC), 2014.

[3] Piotr Doller
    "Piotr抯 Image and Video Matlab Toolbox (PMT)."
    http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html.

Contact:
Di Wu
http://stevenwudi.github.io/
email: stevenwudi@gmail.com  2017/08/01
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import cv2
import os
import h5py
import time
import matlab.engine
import matlab


class bmvc_2014_pami_2014_fDSST:
    def __init__(self,
                 padding=2.0,
                 output_sigma_factor=float(1/16.),
                 scale_sigma_factor=float(1/16.),
                 lambda_value=1e-2,
                 interp_factor=0.025,
                 num_compressed_dim=18,
                 refinement_iterations=1,
                 translation_model_max_area=np.inf,
                 interpolate_response=1,
                 resize_factor=1,
                 number_of_scales=17,
                 number_of_interp_scales=33.,
                 scale_model_factor=1.0,
                 scale_step=1.02,
                 scale_model_max_area=512,
                 s_num_compressed_dim='MAX',
                 feature_ratio=int(4.0),
                 compressed_features='hog_gray',
                 non_compressed_features=None,
                 kernel='linear',
                 w2c_file_path='trackers/w2crs.mat',
                 visualisation=0,
                 matlab_legacy=True,
                 matlab_eng=None,
                 ):
        """
        :param padding: extra area surrounding the target
        :param output_sigma_factor: standard deviation for the desired translation filter output
        :param scale_sigma_factor: standard deviation for the desired scale filter output
        :param lambda_value: regularisation weight (denoted "lambda" in the paper)
        :param interp_factor: tracking model learning rate (denoted "eta" in the paper)
        :param num_compressed_dim: the dimensionality of the compressed features
        :param refinement_iterations: number of iterations used to refine the resulting position in a frame
        :param translation_model_max_area: maximum area of the translation model
        :param interpolate_response: interpolation method for the translation scores
        :param resize_factor: initial resize
        :param number_of_scales: number of scale levels
        :param number_of_interp_scales: number of scale levels after interpolation
        :param scale_model_factor: relative size of the scale sample
        :param scale_step: scale increment factor (denoted 'a" in the paper)
        :param scale_model_max_area: the maximum size of scale examples
        :param s_num_compressed_dim: number of compressed scale feature dimensions
        :param feature_ratio: HOG window size
        :param compressed_features: names of the features for PCA compression
        :param non_compressed_features: not compressed features
        :param kernel: 'linear' or 'rbf'
        :param visualisation: flag for visualistion
        """
        self.padding = padding
        self.output_sigma_factor = output_sigma_factor
        self.scale_sigma_factor = scale_sigma_factor
        self.lambda_value = lambda_value
        self.interp_factor = interp_factor
        self.num_compressed_dim = num_compressed_dim
        self.refinement_iterations = refinement_iterations
        self.translation_model_max_area = translation_model_max_area
        self.interpolate_response = interpolate_response
        self.resize_factor = resize_factor
        self.number_of_scales = number_of_scales
        self.number_of_interp_scales = number_of_interp_scales
        self.scale_model_factor = scale_model_factor
        self.scale_step = scale_step
        self.scale_model_max_area = scale_model_max_area
        self.s_num_compressed_dim = s_num_compressed_dim
        self.visualisation = visualisation
        self.feature_ratio = feature_ratio
        self.compressed_features = compressed_features
        self.non_compressed_features = non_compressed_features
        self.kernel = kernel

        self.old_cov_matrix = []
        self.old_cov_matrix_scale = []
        self.res = []

        if self.compressed_features == 'cn':
            if os.path.isfile(w2c_file_path):
                f = h5py.File(w2c_file_path)
                self.w2c = f['w2crs'].value.transpose(1,0)
            else:
                print("W2C file not found!")
            self.non_compressed_features = 'gray'
            self.compressed_features = compressed_features
            self.num_compressed_dim = 2
            self.compression_learning_rate = 0.15

        if self.kernel == 'gaussian':
            self.sigma = 0.2
            self.learning_rate = 0.075
            self.interp_factor = self.learning_rate

        if self.number_of_scales > 0:
            self.scale_sigma = self.number_of_interp_scales * self.scale_sigma_factor
            self.scale_exp = (np.arange(self.number_of_scales) - np.floor(self.number_of_scales/2)) * \
                             (self.number_of_interp_scales/self.number_of_scales)
            self.scale_exp_shift = np.roll(self.scale_exp, int(-np.floor((self.number_of_scales-1)/2)))

            self.interp_scale_exp = np.arange(self.number_of_interp_scales) - np.floor(self.number_of_interp_scales/2)
            self.interp_scale_exp_shift = np.roll(self.interp_scale_exp, int(-np.floor((self.number_of_interp_scales-1)/2)))

            self.scaleSizeFactors = self.scale_step ** self.scale_exp
            self.interpScaleFactors = self.scale_step ** self.interp_scale_exp_shift

            self.ys = np.exp(-0.5 * self.scale_exp_shift**2 / self.scale_sigma**2)
            self.ysf = np.fft.fft(self.ys)
            self.scale_wnidow = np.hanning(self.ysf.shape[0])

        self.name = 'bmvc_2014_pami_2014_fDSST'
        self.matlab_legacy = matlab_legacy
        if self.matlab_legacy:
            self.eng = matlab_eng
        #     print('starting matlab engine...')
        #     start_time = time.time()
        #     self.eng = matlab.engine.start_matlab("-nojvm -nodisplay -nosplash")
        #     self.eng.addpath('./tools/piotr_hog_matlab')
        #     total_time = time.time() - start_time
        #     print('matlab engine started, used %.2f second'%(total_time))

        self.feature_type = 'fDSST'

    def train(self, im, init_rect):
        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        self.res.append(init_rect)
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.im_sz = im.shape[:2]
        if np.prod(self.target_sz) > self.translation_model_max_area:
            self.currentScaleFactor = np.sqrt(np.prod(self.init_target_sz) / self.translation_model_max_area)
        else:
            self.currentScaleFactor = 1.0
        # target size at the initial scale
        self.init_target_sz = self.target_sz / self.currentScaleFactor
        # window size, taking padding into account
        self.patch_size = np.floor(self.init_target_sz * (1 + self.padding)).astype(int)

        if self.compressed_features == 'gray_hog':
            self.output_sigma = np.sqrt(np.prod(np.floor(self.init_target_sz / self.feature_ratio))) * self.output_sigma_factor
        elif self.compressed_features == 'cn':
            self.output_sigma = np.sqrt(np.prod(self.init_target_sz)) * self.output_sigma_factor

        self.use_sz = np.floor(self.patch_size/self.feature_ratio)
        # compute Y
        grid_y = np.roll(np.arange(np.floor(self.use_sz[0])) - np.floor(self.use_sz[0] / 2), int(-np.floor(self.use_sz[0]/2)))
        grid_x = np.roll(np.arange(np.floor(self.use_sz[1])) - np.floor(self.use_sz[1] / 2), int(-np.floor(self.use_sz[1]/2)))
        rs, cs = np.meshgrid(grid_x, grid_y)
        self.y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = self.fft2(self.y)
        if self.interpolate_response:
            self.interp_sz = np.array(self.y.shape) * self.feature_ratio
            rf = self.resizeDFT_2D(self.yf, self.interp_sz)
            r = np.real(np.fft.ifft2(rf))
            # target location is at the maximum response
            self.v_centre_y, self.h_centre_y = np.unravel_index(r.argmax(), r.shape)

        # store pre-computed cosine window
        self.cos_window = np.outer(np.hanning(self.use_sz[0]), np.hanning(self.use_sz[1]))

        if self.number_of_scales > 0:
            # make sure the scale model is not too large so as to save computation time
            if self.scale_model_factor**2 * np.prod(self.init_target_sz) > self.scale_model_max_area:
                self.scale_model_factor = np.sqrt(float(self.scale_model_max_area)/np.prod(self.init_target_sz))

            # set the scale model size
            self.scale_model_sz = np.floor(self.init_target_sz * self.scale_model_factor).astype(int)
            # force reasonable scale changes
            self.min_scale_factor = self.scale_step ** np.ceil(np.log(np.max(5. / self.patch_size)) / np.log(self.scale_step))
            self.max_scale_factor = self.scale_step ** np.floor(np.log(np.min(np.array([im.shape[0], im.shape[1]]) * 1.0 / self.init_target_sz)) / np.log(self.scale_step))

            if self.s_num_compressed_dim == 'MAX':
                self.s_num_compressed_dim = len(self.scaleSizeFactors)
        ################################################################################################################
        # Compute coefficients for the translation filter
        ################################################################################################################
        # extract the feature map of the local image patch to train the classifer
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size*self.currentScaleFactor)
        # initiliase the appearance
        self.h_num_npca, self.h_num_pca = self.get_features(self.im_crop)

        # if dimensionality reduction is used: update the projection matrix
        # refere to tPAMI paper eq. (7a)
        self.projection_matrix, self.old_cov_matrix = \
            self.calculate_projection(self.h_num_pca, self.num_compressed_dim, self.interp_factor, old_cov_matrix=[])

        # project the features of the new appearance example using the new projection matrix
        self.h_proj = self.feature_projection(self.h_num_npca, self.h_num_pca, self.projection_matrix, self.cos_window)

        if self.kernel == 'linear':
            self.hf_proj = self.fft2(self.h_proj)
            self.hf_num = np.multiply(np.conj(self.yf[:, :, None]), self.hf_proj)
            self.hf_den = np.sum(np.multiply(self.hf_proj, np.conj(self.hf_proj)), 2) + self.lambda_value

        elif self.kernel == 'gaussian':
            # TODO: gaussian kernel
            self.kf = self.fft2(self.dense_gauss_kernel(self.sigma, self.h_proj))
            self.alpha_num = np.multiply(self.yf, self.kf)
            self.alpha_den = np.multiply(self.kf, (self.kf + self.lambda_value))

        ################################################################################################################
        # Compute coefficents for the scale filter
        ################################################################################################################
        if self.number_of_scales > 0:
            self.s_num = self.get_scale_subwindow(im, self.pos, self.init_target_sz, self.currentScaleFactor * self.scaleSizeFactors, self.scale_model_sz)
            # project the features of the new appearance example using the new projection matrix
            self.projection_matrix_scale = self.calculate_projection(self.s_num, self.s_num_compressed_dim)
            # self.s_proj is of dim D * N !
            self.s_proj = self.feature_projection([], self.s_num, self.projection_matrix_scale, self.scale_wnidow)

            if self.kernel == 'linear':
                self.sf_proj = np.fft.fft(self.s_proj, axis=1)
                self.sf_num = np.multiply(self.ysf, np.conj(self.sf_proj))
                self.sf_den = np.sum(np.multiply(self.sf_proj, np.conj(self.sf_proj)), 0)

            elif self.kernel == 'gaussian':
                # TODO: gaussian kernel
                pass

    def detect(self, im, frame):
        if self.kernel == 'gaussian':
            self.zp = self.feature_projection(self.h_num_npca, self.h_num_pca, self.projection_matrix, self.cos_window)

        # extract the feature map of the local image patch to train the classifer
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size * self.currentScaleFactor)

        if self.number_of_scales > 0:
            # for imresize, we always need to pre-cast the image to the desired type
            self.im_crop = imresize(np.array(self.im_crop, dtype=np.uint8), self.patch_size)

        self.xl_npca, self.xl_pca = self.get_features(self.im_crop)
        # project the features of the new appearance example using the new projection matrix
        xt = self.feature_projection(self.xl_npca, self.xl_pca, self.projection_matrix, self.cos_window)

        if self.kernel == 'linear':
            xtf = self.fft2(xt)
            response_f = np.divide(np.sum(np.multiply(np.conj(self.hf_num), xtf), 2), self.hf_den)

        elif self.kernel == 'gaussian':
            self.kf = self.fft2(self.dense_gauss_kernel(self.sigma, xt, self.zp))
            response_f = np.divide(np.multiply(self.alpha_num, self.kf), self.alpha_den)

        # if undersampling the features, we want to interpolate the response so it has the same size as the image patch
        # Wudi's thought; really?
        if self.interpolate_response:
            response_f = self.resizeDFT_2D(response_f, self.interp_sz)

        self.response = np.real(np.fft.ifft2(response_f))
        # target location is at the maximum response
        row, col = np.unravel_index(self.response.argmax(), self.response.shape)
        self.vert_delta = np.mod(row + np.floor((self.interp_sz[0]-1)/2), self.interp_sz[0]) - np.floor((self.interp_sz[0]-1)/2)
        self.horiz_delta = np.mod(col + np.floor((self.interp_sz[1]-1)/2), self.interp_sz[1]) - np.floor((self.interp_sz[1]-1)/2)

        if self.interpolate_response:
            translation_vec = np.array([self.vert_delta, self.horiz_delta]) * self.currentScaleFactor

        else:
            translation_vec = np.array([self.vert_delta, self.horiz_delta]) * self.currentScaleFactor * self.feature_ratio

        #print("frame %d: "%(frame))
        #print(translation_vec)
        self.pos = np.array(self.pos) + translation_vec
        ################################################################################################################
        # update the scale
        ################################################################################################################
        if self.number_of_scales > 0:
            xs_pca = self.get_scale_subwindow(im, self.pos, self.init_target_sz,
                                              self.currentScaleFactor * self.scaleSizeFactors, self.scale_model_sz)
            # project the features of the new appearance example using the new projection matrix of dim D * N !
            xs = self.feature_projection([], xs_pca, self.projection_matrix_scale, self.scale_wnidow)
            xsf = np.fft.fft(xs, axis=1)
            scale_response_f = np.divide(np.sum(np.multiply(self.sf_num, xsf), 0), self.sf_den + self.lambda_value)
            scale_response_f_resize = self.resizeDFT_1D(scale_response_f, self.number_of_interp_scales)
            scale_response = np.real(np.fft.ifft(scale_response_f_resize))
            recovered_scale = np.argmax(scale_response)
            # update the scale
            self.currentScaleFactor *= self.interpScaleFactors[recovered_scale]
            if self.currentScaleFactor < self.min_scale_factor:
                self.currentScaleFactor = self.min_scale_factor
            elif self.currentScaleFactor > self.max_scale_factor:
                self.currentScaleFactor = self.max_scale_factor
            # we only update the target size here.
            self.target_sz = np.multiply(self.currentScaleFactor, self.init_target_sz)
            #print('frame %d, current scale: %.3f'%(frame, self.currentScaleFactor))

        ################################################################################################################
        # Next are for updating the tracker
        # Compute coefficients for the translation filter
        ################################################################################################################
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size * self.currentScaleFactor)
        if self.number_of_scales > 0:
            # for imresize, we always need to pre-cast the image to the desired type
            self.im_crop = imresize(np.array(self.im_crop, dtype=np.uint8), self.patch_size)

        xl_npca, xl_pca = self.get_features(self.im_crop)
        self.h_num_pca = (1 - self.interp_factor) * self.h_num_pca + self.interp_factor * xl_pca
        if len(self.h_num_npca): self.h_num_npca = (1 - self.interp_factor) * self.h_num_npca + self.interp_factor * xl_npca

        # refere to tPAMI paper eq. (7a)
        if self.kernel == 'linear':
            self.projection_matrix, _ = self.calculate_projection(self.h_num_pca, self.num_compressed_dim,
                                                                  self.interp_factor, old_cov_matrix=[])

            self.h_proj = self.feature_projection(self.h_num_npca, self.h_num_pca, self.projection_matrix, self.cos_window)
            self.hf_proj = self.fft2(self.h_proj)
            self.hf_num = np.multiply(np.conj(self.yf[:, :, None]), self.hf_proj)
            # Refer to Eq. (7b)
            xl = self.feature_projection(xl_npca, xl_pca, self.projection_matrix, self.cos_window)
            xlf = self.fft2(xl)
            new_hf_den = np.sum(np.multiply(xlf, np.conj(xlf)), 2) + self.lambda_value
            self.hf_den = (1 - self.interp_factor) * self.hf_den + self.interp_factor * new_hf_den
            # self.x is used for visualisation
            self.x = np.mean(self.h_proj, axis=2)

        elif self.kernel == 'gaussian':
            self.projection_matrix, self.old_cov_matrix = self.calculate_projection(self.h_num_pca, self.num_compressed_dim,
                                                                                    self.compression_learning_rate, self.old_cov_matrix)
            self.x = self.feature_projection(xl_npca, xl_pca, self.projection_matrix, self.cos_window)
            kf_new = self.fft2(self.dense_gauss_kernel(self.sigma, self.x))
            alpha_num_new = np.multiply(self.yf, kf_new)
            alpha_den_new = np.multiply(kf_new, (kf_new + self.lambda_value))
            # subsequence frames, update the model
            self.alpha_num = (1 - self.learning_rate) * self.alpha_num + self.learning_rate * alpha_num_new
            self.alpha_den = (1 - self.learning_rate) * self.alpha_den + self.learning_rate * alpha_den_new

        ################################################################################################################
        # Compute coefficents for the scale filter
        ################################################################################################################
        if self.number_of_scales > 0:
            xs_pca = self.get_scale_subwindow(im, self.pos, self.init_target_sz,
                                              self.currentScaleFactor * self.scaleSizeFactors, self.scale_model_sz)
            self.s_num = (1 - self.interp_factor) * self.s_num + self.interp_factor * xs_pca
            # project the features of the new appearance example using the new projection matrix
            self.projection_matrix_scale = self.calculate_projection(self.s_num, self.s_num_compressed_dim)
            # self.s_proj is of dim D * N !
            self.s_proj = self.feature_projection([], self.s_num, self.projection_matrix_scale, self.scale_wnidow)
            # compute den for scale
            self.projection_matrix_scale_den = self.calculate_projection(xs_pca, self.s_num_compressed_dim)
            xs = self.feature_projection([], xs_pca, self.projection_matrix_scale_den, self.scale_wnidow)

            if self.kernel == 'linear':
                self.sf_proj = np.fft.fft(self.s_proj, axis=1)
                self.sf_num = np.multiply(self.ysf, np.conj(self.sf_proj))

                xsf = np.fft.fft(xs, axis=1)
                new_sf_den = np.sum(np.multiply(xsf, np.conj(xsf)), 0)
                self.sf_den = (1 - self.interp_factor) * self.sf_den + self.interp_factor * new_sf_den

            elif self.kernel == 'gaussian':
                # TODO: gaussian kernel
                pass

        output_rect = [self.pos[1] - self.target_sz[1]/2., self.pos[0] - self.target_sz[0]/2., self.target_sz[1], self.target_sz[0]]
        self.res.append(output_rect)

        return output_rect

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        return np.fft.fft2(x, axes=(0, 1))

    def get_subwindow(self, im, pos, sz):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """

        if np.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0] / 2)
        xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1] / 2)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= self.im_sz[0]] = self.im_sz[0] - 1

        xs[xs < 0] = 0
        xs[xs >= self.im_sz[1]] = self.im_sz[1] - 1

        return im[np.ix_(ys, xs)]

    def get_scale_subwindow(self, im, pos, base_target_sz, scaleFactors, scale_model_sz):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """
        out_pca = []

        for s in range(len(scaleFactors)):
            patch_sz = np.floor(base_target_sz * scaleFactors[s])

            ys = np.floor(pos[0]) + np.arange(patch_sz[0], dtype=int) - np.floor(patch_sz[0] / 2)
            xs = np.floor(pos[1]) + np.arange(patch_sz[1], dtype=int) - np.floor(patch_sz[1] / 2)

            ys = ys.astype(int)
            xs = xs.astype(int)

            # check for out-of-bounds coordinates and set them to the values at the borders
            ys[ys < 0] = 0
            ys[ys >= self.im_sz[0]] = self.im_sz[0] - 1

            xs[xs < 0] = 0
            xs[xs >= self.im_sz[1]] = self.im_sz[1] - 1

            # extract image
            im_patch = im[np.ix_(ys, xs)]
            im_gray = self.rgb2gray(im_patch)

            # extract scale features
            if self.matlab_legacy:
                img_matlab = matlab.single(im_gray.tolist())
                scale_model_sz_matlab = matlab.single(scale_model_sz.tolist())
                img_matlab_resize = self.eng.mexResize(img_matlab, scale_model_sz_matlab, 'auto')
                temp_pca_matlab = self.eng.fhog(img_matlab_resize, 4)
                temp_hog = np.array(temp_pca_matlab._data).reshape(temp_pca_matlab.size[::-1]).T
            else:
                im_patch_resized = imresize(np.array(im_gray, dtype=np.uint8), scale_model_sz.astype(int))
                temp_hog = pyhog.features_pedro(im_patch_resized.astype(np.float64) / 255.0, int(self.feature_ratio))
                neg_idx = temp_hog < 0
                temp_hog[neg_idx] = 0
            out_pca.append(temp_hog[:, :, :31].flatten(order='F'))

        return np.asarray(out_pca, dtype='float')

    def get_features(self, im_crop):
        """
        :param im_crop:
        :return:
        """
        # because the hog output is (dim/4)>1:
        # if self.patch_size.min() < 12:
        #     scale_up_factor = 12. / np.min(im_crop)
        #     im_crop = imresize(np.array(im_crop, dtype=np.uint8), np.asarray(self.patch_size * scale_up_factor).astype('int'))
        xo_npca, xo_pca = [], []

        im_gray = self.rgb2gray(im_crop)
        cell_gray = self.cell_gray(im_gray)

        if self.non_compressed_features == 'gray':
            xo_npca = im_gray /255. - 0.5

        if self.compressed_features == 'cn':
            xo_pca_temp = self.im2c(im_crop)
            xo_pca = np.reshape(xo_pca_temp,
                                (np.prod([xo_pca_temp.shape[0], xo_pca_temp.shape[1]]), xo_pca_temp.shape[2]))

        elif self.compressed_features == 'gray_hog':
            if self.matlab_legacy:
                img_matlab = matlab.single(im_gray.tolist())
                temp_pca_matlab = self.eng.fhog(img_matlab, 4)
                features_hog = np.array(temp_pca_matlab._data).reshape(temp_pca_matlab.size[::-1]).T
            else:
                features_hog = pyhog.features_pedro(im_crop.astype(np.float64) / 255.0, int(self.feature_ratio))
                neg_idx = features_hog < 0
                features_hog[neg_idx] = 0

            temp_pca = np.concatenate([features_hog[:, :, :31], cell_gray[:, :, np.newaxis]], axis=2)
            xo_pca = temp_pca.reshape([temp_pca.shape[0]*temp_pca.shape[1], temp_pca.shape[2]], order='F')

        return xo_npca, xo_pca

    def cell_gray(self, gray_img):
        """
        Average the intensity over a single hog-cell
        :param img:
        :return:
        """
        # compute the integral image
        integral_img = cv2.integral(gray_img)
        cell_size = int(self.feature_ratio)

        i1 = np.array(range(cell_size, gray_img.shape[0]+1, cell_size))
        i2 = np.array(range(cell_size, gray_img.shape[1]+1, cell_size))
        A1, A2 = np.meshgrid(i1 - cell_size, i2 - cell_size)
        B1, B2 = np.meshgrid(i1, i2 - cell_size)
        C1, C2 = np.meshgrid(i1 - cell_size, i2)
        D1, D2 = np.meshgrid(i1, i2)

        cell_sum = integral_img[A1, A2] - integral_img[B1, B2] - integral_img[C1, C2] + integral_img[D1, D2]
        cell_gray = cell_sum.T / (cell_size**2 * 255) - 0.5

        return cell_gray

    def calculate_projection(self, z_pca, num_compressed_dim, compression_learning_rate=1, old_cov_matrix=[]):
        # we use SVD if feature dimensionality is smaller than the number of elements :
        # this is true for the translation filter
        if z_pca.shape[0] > z_pca.shape[1]:
            # compute the mean appearance
            # data_mean = np.mean(z_pca, axis=0)
            # # substract the mean from the appearance to get the data matrix
            # data_matrix = np.subtract(z_pca, data_mean[None, :])
            # # calculate the covariance matrix
            # cov_matrix = np.cov(data_matrix.T)
            # # calculate the principal components (pca_basis) and corresponding variances
            # if len(old_cov_matrix):
            #     cov_matrix = (1 - compression_learning_rate) * old_cov_matrix + \
            #                  compression_learning_rate * cov_matrix
            # else:
            #     cov_matrix = cov_matrix

            cov_matrix = np.dot(z_pca.T, z_pca)
            U, s, V = np.linalg.svd(cov_matrix)
            S = np.diag(s)

            # calculate the projection matrix as the first principal components
            # and extract their corresponding variances
            projection_matrix = U[:, :num_compressed_dim]
            projection_variances = S[:num_compressed_dim, :num_compressed_dim]
            # initialise the old covariance matrix using the computed projection matrix and variance
            if len(old_cov_matrix):
                old_cov_matrix = (1 - compression_learning_rate) * old_cov_matrix \
                                      + compression_learning_rate * \
                                        np.dot(np.dot(projection_matrix, projection_variances), projection_matrix.T)
            else:
                old_cov_matrix = np.dot(np.dot(projection_matrix, projection_variances), projection_matrix.T)

            return projection_matrix, old_cov_matrix
        # for scale filter, the feature dimensionality is larger than the number of element
        # we use QR factorisation which is both computational and memory efficient
        # and we do not explicitly construct the auto-correlation matrix
        else:
            scale_basis, _= np.linalg.qr(z_pca.T, 'full')
            projection_matrix = scale_basis.T
            return projection_matrix

    def feature_projection(self, xo_npca, xo_pca, projection_matrix, cos_window):
        """
        Calcaultes the compressed feature map by mapping the PCA features with the projection matrix
        and concatinates this with the non-PCA features. The feature map is than multiplied with a cosine-window.
        :return:
        """
        if not self.compressed_features:
            z = xo_npca
        else:
            # project the PCA-features using the projection matrix and reshape to a window
            if len(cos_window.shape) > 1:
                x_proj_pca = np.dot(xo_pca, projection_matrix).reshape(
                    (cos_window.shape[0], cos_window.shape[1], projection_matrix.shape[1]), order='F')
            else:
                # one dimensional scale featurers
                x_proj_pca = np.dot(projection_matrix, xo_pca.T)

            # concatinate the feature windows
            if not self.non_compressed_features:
                z = x_proj_pca
            else:
                # gray scale concatenate with color names
                if len(xo_npca.shape) != len(x_proj_pca.shape):
                    z = np.concatenate((xo_npca[:, :, None], x_proj_pca), axis=2)

        if len(cos_window.shape) > 1:
            features = np.multiply(z, cos_window[:, :, None])
        else:
            features = np.multiply(z, cos_window[None, :])

        return features

    def resizeDFT_2D(self, inputdft, desiredSize):
        """
        Refer to tPAMI paper sect: 5.2.1 sub-grid interpolation of correlation scores
        Sub-grid interpolation allows us to use coarser feature grids for the training and detection samples.
        This affects the computational cost by reducing the size of the performed FFTs required for training and
        detection respectively. We employ interpolation with trigonometric polynomials [*]. This is especially suitable
        since the DFT coefficients of the correlation score, required to  perform the interpolation, are already
        computed in. The interpolated scores y_t are obtained by zero-padding the high frequencies of Y_t in such that
        its size is equal to the size of the interpolation grid. The interpolated score y_t are then obtain by
        performing the inverse DFT of the padded Y_t.

        [*] A. V. Oppenheim, A. S. Willsky, and S. H. Nawab, Signals &Amp; Systems (2Nd Ed.). Upper Saddle River, NJ,
        USA: Prentice-Hall, Inc., 1996.]
        :param inputdft:
        :param desiredSize:
        :return:
        """
        imsz = np.array(inputdft.shape)
        minsz = np.minimum(imsz, np.array(desiredSize).astype(int))
        scaling = np.prod(desiredSize) / np.prod(imsz)

        resized_dft = np.zeros(shape=np.array(desiredSize).astype(int), dtype='complex')

        mids = np.array(np.ceil(minsz/2.), dtype=int)
        mide = np.array(np.floor((minsz-1)/2.), dtype=int)

        resized_dft[:mids[0], :mids[1]] = scaling * inputdft[:mids[0], :mids[1]]
        resized_dft[:mids[0], -mide[1]:] = scaling * inputdft[:mids[0], -mide[1]:]
        resized_dft[-mide[0]:, :mids[1]] = scaling * inputdft[-mide[0]:, :mids[1]]
        resized_dft[-mide[0]:, -mide[1]:] = scaling * inputdft[-mide[0]:, -mide[1]:]

        return resized_dft

    def resizeDFT_1D(self, input_dft_1d, desired_len):

        length = len(input_dft_1d)
        minsz = np.minimum(length, desired_len)
        scaling = float(desired_len) / length

        resized_dft = np.zeros(shape=int(desired_len), dtype='complex')
        mids = int(np.ceil(minsz/2))
        mide = int(np.floor((minsz-1)/2))

        resized_dft[:mids] = scaling * input_dft_1d[:mids]
        resized_dft[-mide:] = scaling * input_dft_1d[-mide:]

        return resized_dft

    def im2c(self, im):
        """
        Calcalate Color Names according to the paper:
        [3] J. van de Weijer, C. Schmid, J. J. Verbeek, and D. Larlus.
        "Learning color names for real-world applications."
        TIP, 2009
        :param im:
        :return:
        """
        RR = im[:, :, 0]
        GG = im[:, :, 1]
        BB = im[:, :, 2]

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
        # 'F' , means to flatten in column-major
        # Because we convert w2c from matlab which is a column-major programming language, Duh >:<
        index_im = np.floor(np.ndarray.flatten(RR, 'F') / 8.) + \
                   32 * np.floor(np.ndarray.flatten(GG, 'F') / 8.) + \
                   32 * 32 * np.floor(np.ndarray.flatten(BB, 'F') / 8.)

        out = np.reshape(self.w2c[index_im.astype('int')], (im.shape[0], im.shape[1], self.w2c.shape[1]), 'F')
        return out

    def dense_gauss_kernel(self, sigma, x, y=None):
        """
        Gaussian Kernel with dense sampling.
        Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
        between input images X and Y, which must both be MxN. They must also
        be periodic (ie., pre-processed with a cosine window). The result is
        an MxN map of responses.

        If X and Y are the same, ommit the third parameter to re-use some
        values, which is faster.
        :param sigma: feature bandwidth sigma
        :param x:
        :param y: if y is None, then we calculate the auto-correlation
        :return:
        """
        N = np.prod(x.shape)
        xf = self.fft2(x)
        xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

        if y is None:
            # auto-correlation of x
            yf = xf
            yy = xx
        else:
            yf = self.fft2(y)
            yy = np.dot(y.flatten().transpose(), y.flatten())  # squared norm of y

        xyf = np.multiply(xf, np.conj(yf))
        xy = np.real(np.fft.ifft2(np.sum(xyf, axis=2)))

        k = np.exp(-1. / sigma ** 2 * np.maximum(0, (xx + yy - 2 * xy)) / N)

        return k

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])