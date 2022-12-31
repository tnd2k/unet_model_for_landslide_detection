class pp():

    def find_data(local_dir):
        '''
        *Only use for H5PY data
        This function will help to find the data in the local_dir folder variable and return the filepaths list and list of open H5PY data
        Requirements: h5py, glob
        
        Returns: filepaths list and open H5PY list
        '''
        import os
        import glob
        import h5py
        import time
        
        init_time = time.time()

        if not os.path.exists(local_dir):
            raise Exception('The folder path does not exist! Please check the path again.')

        image_files = glob.glob(f'{local_dir}/*.h5',recursive = True)
        image_holder=[]

        # reading file
        for file in image_files:
            # read file
            f = h5py.File(file)
            image_holder.append(f)

        final_time = time.time()    
        tot_time = final_time-init_time
        
        print(f'Files are found in folder {local_dir} in {tot_time} seconds')

        return image_files, image_holder

    def h5label_to_array(label_holder):
        '''
        This function reads array of labeling from the open H5PY list
        Args: 
            label_holder: your open H5PY label file list
        
        Requirements: time, tqdm
        
        Returns: array-like list
        '''
        import time
        from tqdm.notebook import tqdm
        
        init_time = time.time()
        
        for key in label_holder[0].keys():
            k = key
        
        print('Extracting...')
        raw_label = []
        for i in tqdm(range(len(label_holder)), total=len(label_holder), desc='Progress'):
            raw_label.append(label_holder[i][k][:,:])
        
        final_time = time.time()
        tot_time = final_time - init_time
        
        print(f'Convert h5py to numpy array done in {tot_time} seconds')
        
        return raw_label
    
    
    def rgb_norm_data(holder_var, data_type):
        '''
        * Only use for imagery data
        This function will normalize the data with float input data and return the normalized data list
        And the holder-list variable of opening H5PY data will be shut down to free RAM space
        The method is that the Red, Green, Blue bands will be separated and divided by the maximun value of that band. Then, these bands are stacked again
        However, you need to specify the band order based on imagery platform: possible float32, rgb-scale (0-255), ...
        Requirements: numpy
        
        Args:
            holder_var: your array variables
            data_type (options): float32, rgb-scale
        
        Returns: normalization-like array
        '''

        import numpy as np
        from tqdm.notebook import tqdm
        import time
        
        holder_norms = [] # list to hold all normalized data
        
        print('Start normalization...')
        init_time = time.time()
        for i in tqdm(range(len(holder_var)), total=len(holder_var), desc='Normalization'):
            if data_type == 'float32':
                red = holder_var[i]['img'][:,:,3]/np.max(holder_var[i]['img'][:,:,3])
                green = holder_var[i]['img'][:,:,2]/np.max(holder_var[i]['img'][:,:,2])
                blue = holder_var[i]['img'][:,:,1]/np.max(holder_var[i]['img'][:,:,1])

            elif data_type == 'rgb-scale':
                red = holder_var[i]['img'][:,:,0]/255
                green = holder_var[i]['img'][:,:,1]/255
                blue = holder_var[i]['img'][:,:,2]/255

            holder_norm = np.dstack((red, green, blue))
            holder_norms.append(holder_norm)

        # check if shape of channel, height, width
        if holder_norms[0].shape[0] <= 3:
            holder_norms = np.moveaxis(holder_norms,0,-1) # re-order the shape
            print('Channel-goes-first (channel, height, width) was moved to form channel-goes-last (height, width, channel)')
        else:
            print('Channel-goes-last (height, width, channel)')

        # close the open h5py files
        print('Free RAM space')
        for items in tqdm(holder_var, desc='Loading...'):
            items.close()
            
        final_time = time.time()
        tot_time = final_time - init_time
        print(f'Normalization is processed in {tot_time} seconds')
        
        return holder_norms


    def flip_image(images, masks, axis_to_flip=None):
        '''
        This function returns a single array with side-flipped tuples
        
        Args:
            images: your normalized image array variable
            masks: your open H5PY label array variable
            axis_to_flip:
                None: default (0,1)
                0: Flip vertically
                1: Flip horizontally
            
        Returns: flipped_array
        '''
        import numpy as np
        from tqdm.notebook import tqdm
        import time
        
        init_time = time.time()
        
        flipped_images = []
        flipped_masks = []
        
        # find key of mask data
        for key in masks[0].keys():
            k = key
        print('Flipping...')
        
        for i in tqdm(range(len(images)), total=len(images), desc='In Flipping loop'):
            flipped_images.append(np.flip(images[i], axis=axis_to_flip))
            flipped_masks.append(np.flip(masks[i][k][:,:], axis=axis_to_flip))
            
        final_time = time.time()
        tot_time = final_time - init_time
        
        print(f'Flipping done in {tot_time} seconds')
        
        return flipped_images, flipped_masks


    def rotate_image(images, masks, a, reshape=False):
        '''
        This function returns a single array with side-rotated tuples
        
        Args:
            images: your normalized array variable
            masks: your label holder list variable
            a: angle, int
            reshape: default to False, as you wish to keep array's shape
        
        Returns: rotated_array
        '''
        from scipy.ndimage import rotate
        from tqdm.notebook import tqdm
        import time
        
        # raise error before running any task
        if '.' in str(a):
            console.alert('The angle must be an integer number!')
        else:
            pass
        
        init_time = time.time()
        print('Rotating...')
        
        # find key of mask data
        for key in masks[0].keys():
            k = key
        rotated_image = []
        rotated_mask = []
        for i in tqdm(range(len(images)), total=len(images), desc='In Rotating loop'):
            rotated_image.append(rotate(images[i], angle=a, reshape=reshape))
            rotated_mask.append(rotate(masks[i][k][:,:], angle=a, reshape=reshape))

        final_time = time.time()
        tot_time = final_time - init_time
        
        print(f'Rotating done in {tot_time} seconds')
            
        return rotated_image, rotated_mask


    def trim_array(images_data, masks_data, image_size):
        '''
        This easy function that can process your data augmentation step by clip the image to an roi of image_size tile
        Requirements: numpy, random
        
        Args:
            images_data: your array variable, must be a list of array, not just array
            masks_data: your array variable, must be a list of array, not just array
            image_size: dtype of int, must be an integer number
        
        Returns: resized-like array
        '''
        import numpy as np
        import random
        from tqdm.notebook import tqdm
        import time
        
        if images_data[0].shape[0:2] != masks_data[0][:,:].shape:
            raise Exception('images_data and masks_data must have the same shape')
        
        # pad image and mask to ensure image is under control
        print('Padding...')
        
        init_time = time.time()
        
        pad_size = image_size/2
        
        padding_masks = []
        padding_images = []
        for i in tqdm(range(len(images_data)), total=len(images_data), desc='In Padding loop'):
            padding_images.append(np.pad(images_data[i],((int(pad_size),int(pad_size)),(int(pad_size),int(pad_size)),(0,0)),'constant'))
            padding_masks.append(np.pad(masks_data[i],((int(pad_size),int(pad_size)),(int(pad_size),int(pad_size))),'constant'))

        # trim
        print('Trimming...')
        resized_images = []
        resized_masks = []
        for i in tqdm(range(len(padding_images)), total=len(padding_images), desc='In Trimming loop'):
            # params to control the roi to trim array
            x_start=int(round(image_size*random.uniform(0, 0.5)))
            y_start=int(round(image_size*random.uniform(0, 0.5)))
            x_end=int(x_start+image_size)
            y_end=int(y_start+image_size)

            resized_masks.append(padding_masks[i][x_start:x_end,y_start:y_end])
            resized_images.append(padding_images[i][x_start:x_end,y_start:y_end,:])
        
        final_time = time.time()
        tot_time = final_time - init_time
        
        print(f'Trimming done in {tot_time} seconds')
        
        return resized_images, resized_masks


    def filename_split(folder_path):
        '''
        This function will return the path list containing file's name of each file in the provided folder path

        Requirements: os

        Return: a list 
        '''

        import os

        # export index to folder
        path_list = [] # list to hold train files' names

        # split file name
        tfile = os.listdir(folder_path)
        for f in tfile:
            # split text to get the order of the training file
            if '_' in f:
                file_split_text = f.split('_')
            else:
                file_split_text = f
            if '.' in f:
                file_split_tag = file_split_text[-1].split('.')
            else:
                file_split_tag = file_split_text[-1]

            file_name = file_split_tag[0]
            path_list.append(file_name)

        return path_list

class model_fetch():
    
    def generate_tensor_input(parent_dir, imagepatch_filename, labelpatch_filename):
        '''
        This function prepares the necessary data for creating dense tensor input in Encode-Decode step
        Args:
            parent_dir: the folderpath contains the pre-processing numpy arrays
            imagepatch_filename: name of the pre-processing numpy arrays as image
            labelpatch_filename: name of the pre-processing numpy arrays as label
            nclass (int): the number of classes that needs classifying
        
        Requirements: tensorflow, numpy, os
        
        Returns: pre-processing tensor input
        '''
        import numpy as np
        import os
        from tensorflow.keras.layers import Input
        import time
        
        init_time = time.time()

        # load in data
        f_image = os.path.join(parent_dir, imagepatch_filename)
        f_label = os.path.join(parent_dir, labelpatch_filename)
        
        image = np.load(f_image)
        label = np.load(f_label)

        # define inputs
        channel = image.shape[-1]
        img_rows, img_cols = image.shape[1], image.shape[2]
        input_shape = (img_rows, img_cols, channel)
        inputs = Input((input_shape))
        
        final_time = time.time()
        
        tot_time = final_time - init_time
        print(f'Create tensor input done in {tot_time} seconds')
        
        return image, label, inputs
    
    def train_test_split(image, label, train_size, test_size, nclass):
        '''
        This function split the image and label data into train and test dataset to feed the model, and to reduce the input capacity to GPU in case you have low capaciy GPU
        Returns: train_image, train_label, test_image, test_label
        '''
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        
        train_image, test_image, train_label, test_label = train_test_split(image, label, train_size=train_size, test_size=test_size, random_state=123)
        
        # classify label into 0,1 classes
        train_label = tf.keras.utils.to_categorical(train_label,num_classes=nclass, dtype='uint8')
        test_label = tf.keras.utils.to_categorical(test_label,num_classes=nclass, dtype='uint8')
        
        # check dim
        print(f'Dims of train_image: {train_image.shape}')
        print(f'Dims of train_label: {train_label.shape}')
        print(f'Dims of test_image: {test_image.shape}')
        print(f'Dims of test_label: {test_label.shape}')
        
        return train_image, train_label, test_image, test_label

    def model_generation(inputs, nclass):
        '''
        This function uses encode-decode structure to clarify the model, generate the densed tensor inputs into the model
        Args:
            inputs: tensor array
            nclass: the number of classes that need classifying
            
        Returns: model
        '''
        import time
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization
        
        init_time = time.time()
        # encoder
        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(1, 1))(conv1)
        conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)
        conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(1, 1))(conv3)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(1, 1))(conv4)

        # decoder
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(1,1), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(1, 1), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(1, 1), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(1, 1), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)
        conv10 = Conv2D(nclass, (1, 1), activation='sigmoid')(conv9)
        model = Model(inputs=[inputs], outputs=[conv10])
        
        final_time = time.time()
        tot_time = final_time - init_time
        
        model.summary()
        print(f'Encode-Decode done in {tot_time} seconds')
        
        return model

    def train_model(prepared_model, train_image, train_label, test_image, test_label, epochs, batch_size, learning_rate, rho, momentum, metrics):
        '''
        By default, this function uses binary crossentropy loss with RMS optimizer inside the model, parameters by defaults:
            epochs: 300
            batch_size: 30
            learning_rate : 0.001
            rho : 0.9
            momentum (decay) : 0.0
            metrics: ['accuracy']
        Args:
            model: obtained from model_generation function
            image: image stack
            label: label stack
            predict_set: validation stack
            
        Returns: model and its history
        '''
        import os
        import tensorflow as tf
        import time
        import pandas as pd
        
        # start
        init_time = time.time()

        # initialize model
        SGD = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, momentum=momentum)
        catog = tf.keras.losses.binary_crossentropy
        
        # model compilation
        prepared_model.compile(loss = catog, optimizer = SGD, metrics = metrics)
        
        if not os.path.exists('Output/Model/Checkpoints/'):
            try:
                os.mkdir('Output/Model/Checkpoints/')
            except:
                pass
        
        # create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='Output/Model/Checkpoints/cp.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)
        #stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)

        # training by now
        model_history = prepared_model.fit(train_image, train_label, batch_size=batch_size, epochs=epochs, validation_data = (test_image, test_label), callbacks=[cp_callback], shuffle=True)
        
        # convert the history.history dict to a pandas DataFrame:     
        model_history_df = pd.DataFrame(model_history.history) 
        
        # save model history
        hist_csv_file = 'Output/Model/history.csv'
        with open(hist_csv_file, mode='w') as f:
            model_history_df.to_csv(f)
        f.close()

        # save model
        if not os.path.exists('Output/Model/'):
            try:
                os.mkdir('Output/Model/')
            except:
                pass
        prepared_model.save('Output/Model/unet.h5')
        if not os.path.exists('Output/Model/Checkpoints/'):
            try:
                os.mkdir('Output/Model/Checkpoints/')
            except:
                pass
        prepared_model.save_weights('Output/Model/Checkpoints/model_weights.h5')
        
        # final evaluation of the model
        scores = prepared_model.evaluate(test_image, test_label, batch_size=batch_size)

        final_time = time.time()
        tot_time = final_time - init_time
        
        print ('running time: '+ str(tot_time))
        print ('evaluation score: ' + str(scores))
        
        return prepared_model, model_history

    def _predict_(model, predict_set, batch_size, save_predict):
        '''
        Predict!!!
        Args:
            model: the trained model
            predict_set: data for predicting (accept file path or direct data array)
            batch_size: samples of each step
            save_predict: the output path for saving the predictions
        
        Returns: predict dataset, valid images
        '''
        import numpy as np
        import time
        
        init_time = time.time()
        
        if type(predict_set)==str:
            print('Predicting dataset loading...')
            valid = np.load(predict_set)
            # predict by valid set
            print('Predicting...')
            predict = model.predict(valid, batch_size=batch_size)
        else:
            # predict by valid set
            print('Predicting...')
            predict = model.predict(predict_set, batch_size=batch_size)

        # save results
        np.save(save_predict, predict)
        
        final_time = time.time()
        tot_time = final_time - init_time
        print(f'Prediction done in {tot_time} seconds')
        
        return valid, predict
    
    def get_evaluation_scores(predicts, test_labels, save_cm, save_report):
        '''
        This function calculates the precision, accuracy, and f1-score
        '''
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score
        
        # check type of variables
        if type(test_labels)==str:
            test_set = np.load(test_labels)
            test_set = test_set.flatten()
        else:
            test_set = test_labels.flatten()
            
        if type(predicts)==str:
            predictions = np.load(predicts)
        else:
            predictions = predicts
            
        # update predict set
        predictions[(predictions>=0.35)] = 1
        predictions[(predictions<0.35)] = 0
        
        predictions = predictions.flatten()
        
        # scoring
        confusion_matrix = confusion_matrix(y_true=test_set, y_pred=predictions)
        precision = precision_score(y_true=test_set, y_pred=predictions)
        f1_score = f1_score(y_true=test_set, y_pred=predictions)
        accuracy = accuracy_score(y_true=test_set, y_pred=predictions)
        
        # save results
        np.save(save_cm,confusion_matrix)
        with open(save_report,'w') as f:
            f.write(f'confusion matrix: {confusion_matrix}\nprecision: {precision}\nf1-score: {f1_score}\naccuracy: {accuracy}')
        f.close()
        
        print(f'The model is worthy of {accuracy*100} % predicting with a precision of {precision*100}.\n The representative number F1-Score of the model is {f1_score*100}')
        
        return confusion_matrix, precision, f1_score, accuracy

class plots():
    
    def show_direct_metrics(model, epochs, save_folder:str, savefig=True):
        '''
        Returns line plots of accuracy and loss of the model
        '''
        
        # plot model assessment
        import matplotlib.pyplot as plt
        import numpy as np
        # initialize frame
        fig, axs = plt.subplots(2,sharex=True)
        x_axis = np.arange(epochs)
        
        # model accuracy
        axs[0].plot(x_axis,model_history.history['accuracy'])
        axs[0].plot(x_axis,model_history.history['val_accuracy'])
        #axs[0].plot(x_axis,fitting_rate)
        axs[0].set_title('model accuracy')
        axs[0].set_ylabel('accuracy')
        axs[0].set_ylim(0.7,1)
        axs[0].legend(['training', 'validation'], loc='upper left')

        # model loss
        axs[1].plot(x_axis,model_history.history['loss'])
        axs[1].plot(x_axis,model_history.history['val_loss'])
        axs[1].set_title('model loss')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['training', 'validation'], loc='upper left')

        # save plot
        if savefig==True:
            plt.savefig(save_folder+'/metrics.png', dpi=300)
        
        plt.show()
        
    def show_later_metrics(data_path:str, epochs, save_folder:str, savefig=True):
        '''
        Returns line plots of accuracy and loss of the model
        Args:
            data_path: str (csv file)
            epochs: number of training steps
            save_folder: str (folder for output png)
        '''
        
        # plot model assessment
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # read data
        df = pd.read_csv(data_path)
        
        # initialize frame
        fig, axs = plt.subplots(2,sharex=True)
        x_axis = np.arange(epochs)
        
        # model accuracy
        axs[0].plot(x_axis,df['accuracy'])
        axs[0].plot(x_axis,df['val_accuracy'])
        #axs[0].plot(x_axis,fitting_rate)
        axs[0].set_title('model accuracy')
        axs[0].set_ylabel('accuracy')
        axs[0].set_ylim(0.7,1)
        axs[0].legend(['training', 'validation'], loc='upper left')

        # model loss
        axs[1].plot(x_axis,df['loss'])
        axs[1].plot(x_axis,df['val_loss'])
        axs[1].set_title('model loss')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['training', 'validation'], loc='upper left')
        
        # save plot
        if savefig==True:
            plt.savefig(save_folder+'/metrics.png', dpi=300)
        
        plt.show()
        
    def plot_confusion_matrix(cm, classes: list, normalize=False, title='Confusion matrix'):
        '''
        This function plots the confusion matrix
        Args:
            cm: the confusion matrix result
            classes: the ground truth labels (typically, which is the test labels)
        '''
        import numpy as np
        import itertools
        import matplotlib.pyplot as plt
        
        if type(cm) == str:
            cm = np.load(cm)
        else:
            cm = cm
        
        # plot
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
class for_test():
    
    def patching(in_path, input_filename, patch_size):
        '''
        Creates patches from large image
        Args:
            in_path : the folder contains the large image
            input_filename: the filename that is the image needs patching
            path_size: square-sized, e.g. 128 or 256 or 512,...
        '''
        from osgeo import gdal
        from tqdm.notebook import tqdm
        import time
        
        init_time - time.time()
        
        # read band for getting dimensions
        ds = gdal.Open(in_path + input_filename)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        # iterate the image
        for i in tqdm(range(0, xsize + 2*patch_size - xsize%patch_size, patch_size), total=len(0, xsize + 2*patch_size - xsize%patch_size, patch_size), desc="Loop for rows"):
            for j in tqdm(range(0, ysize + 2*patch_size - ysize%patch_size, patch_size), total=len(0, ysize + 2*patch_size - ysize%patch_size, patch_size), desc="Loop for columns"):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(patch_size) + ", " + str(patch_size) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                os.system(com_string)
        
        final_time = time.time()
        tot_time = final_time - init_time
        print(f'Separation done in {tot_time} seconds!')      

    def normalize_patches(img_pred_paths):
    
        # normalize patches
        
        import rasterio as rio
        import numpy as np
        import time
        from tqdm.notebook import tqdm
        
        init_time = time.time()

        img_norms = []
        meta_out = []
        for i in tqdm(range(len(img_pred_paths)), total=len(img_pred_paths), desc="Normalizing..."):
            with rio.open(img_pred_paths[i]) as f:
                img = f.read()

                if img.shape == (img.shape[0], 128,128):
                    # check if the shape of the image starts with number of bands - (bands, height, width), if so, change it to (height, width, bands):
                        img_swap_bands_1 = np.swapaxes(img, 0, 1)
                        img_swap_bands_2 = np.swapaxes(img_swap_bands_1, 1, 2)

                meta = f.profile
                meta.update(count=1)

                img_norms.append(img_swap_bands_2[:,:,:] / 255)

                meta_out.append(meta)
                
        stacked_norms = np.stack(img_norms)
                
        final_time = time.time
        tot_time = final_time - init_time
        
        print(f'Normalization done for patches in {tot_time} seconds!')
        
        return stacked_norms, meta_out

    
    def save_prediction(predict_results, img_pred_paths):
        
        import rasterio as rio
        import numpy as np
        from tqdm.notebook import tqdm
        import time
        
        init_time = time.time()
        
        # save the predicted results
        for i in tqdm(range(len(predict_results)), total=len(predict_results), desc="Saving..."):
            name = img_pred_paths[i].split('Patches')[0]+'Output/'+img_pred_paths[i].split('Patches')[-1].split('\\')[0]+'/'+img_pred_paths[i].split('\\')[-1]
            with rio.open(name, "w", **meta_out[i]) as dest:
                # reclassify the predicted result
                result = np.argmax(predict_results[i])
                dest.write(result[:,:,1], indexes=1)
            dest.close()
        final_time = time.time()
        tot_time = final_time - init_time
        print(f'Saving predicted patches done in {tot_time} seconds')
        
    def mosaicking_pred(pre_re_paths):
        '''
        Args:
            pre_re_paths: folder contains predicting patches
        '''
        
        # merge predicted results
        import glob
        from rasterio.merge import merge
        import time
        
        init_time = time.time()

        src_files_to_mosaic = [] # list for opening rasters

        # open rasters
        for src in pre_re_paths:
            img = rio.open(src)
            src_files_to_mosaic.append(img)

        # copy the metadata
        out_meta = img.meta.copy()

        mosaic, out_trans = merge(src_files_to_mosaic) # merge predicted results

        # write the mosaic raster to disk
        # update the metadata
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": img.crs,
                         "dtype": rio.uint8,
                         "nodata": 0
                         }
                        )

        with rio.open(pred_dir+'Test/Output/MCC/AnhTuyen/Mosaic/pred_BDA_MuCangChai_GeoLLWGS84_P3.tif', "w", **out_meta) as dest:
            dest.write(mosaic)

        dest.close()
        
        final_time = time.time()
        tot_time = final_time - init_time
        print(f"Mosacking done in {tot_time} seconds")






