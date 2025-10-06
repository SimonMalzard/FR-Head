import numpy as np
import sys
import cv2

from torch.utils.data import Dataset

from feeders import tools

ntu120_class_name = [
    "A1. drink water", "A2. eat meal/snack", "A3. brushing teeth", "A4. brushing hair", "A5. drop", "A6. pickup",
    "A7. throw", "A8. sitting down", "A9. standing up (from sitting position)", "A10. clapping", "A11. reading",
    "A12. writing", "A13. tear up paper", "A14. wear jacket", "A15. take off jacket", "A16. wear a shoe",
    "A17. take off a shoe", "A18. wear on glasses", "A19. take off glasses", "A20. put on a hat/cap",
    "A21. take off a hat/cap", "A22. cheer up", "A23. hand waving", "A24. kicking something", "A25. reach into pocket",
    "A26. hopping (one foot jumping)", "A27. jump up", "A28. make a phone call/answer phone", "A29. playing with phone/tablet",
    "A30. typing on a keyboard", "A31. pointing to something with finger", "A32. taking a selfie", "A33. check time (from watch)",
    "A34. rub two hands together", "A35. nod head/bow", "A36. shake head", "A37. wipe face", "A38. salute", "A39. put the palms together",
    "A40. cross hands in front (say stop)", "A41. sneeze/cough", "A42. staggering", "A43. falling", "A44. touch head (headache)",
    "A45. touch chest (stomachache/heart pain)", "A46. touch back (backache)", "A47. touch neck (neckache)", "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)/feeling warm", "A50. punching/slapping other person", "A51. kicking other person",
    "A52. pushing other person", "A53. pat on back of other person", "A54. point finger at the other person",
    "A55. hugging other person", "A56. giving something to other person", "A57. touch other person's pocket",
    "A58. handshaking", "A59. walking towards each other", "A60. walking apart from each other", "A61. put on headphone",
    "A62. take off headphone", "A63. shoot at the basket", "A64. bounce ball", "A65. tennis bat swing", "A66. juggling table tennis balls",
    "A67. hush (quite)", "A68. flick hair", "A69. thumb up", "A70. thumb down", "A71. make ok sign", "A72. make victory sign",
    "A73. staple book", "A74. counting money", "A75. cutting nails", "A76. cutting paper (using scissors)", "A77. snapping fingers",
    "A78. open bottle", "A79. sniff (smell)", "A80. squat down", "A81. toss a coin", "A82. fold paper", "A83. ball up paper",
    "A84. play magic cube", "A85. apply cream on face", "A86. apply cream on hand back", "A87. put on bag", "A88. take off bag",
    "A89. put something into a bag", "A90. take something out of a bag", "A91. open a box", "A92. move heavy objects", "A93. shake fist",
    "A94. throw up cap/hat", "A95. hands up (both hands)", "A96. cross arms", "A97. arm circles", "A98. arm swings", "A99. running on the spot",
    "A100. butt kicks (kick backward)", "A101. cross toe touch", "A102. side kick", "A103. yawn", "A104. stretch oneself",
    "A105. blow nose", "A106. hit other person with something", "A107. wield knife towards other person",
    "A108. knock over other person (hit with body)", "A109. grab other person’s stuff", "A110. shoot at other person with a gun",
    "A111. step on foot", "A112. high-five", "A113. cheers and drink", "A114. carry something with other person",
    "A115. take a photo of other person", "A116. follow other person", "A117. whisper in other person’s ear",
    "A118. exchange things with other person", "A119. support somebody with hand", "A120. finger-guessing game (playing rock-paper-scissors)"
]

ntu120_class_name_short = [
    "A1. drink water", "A2. eat meal", "A3. brushing teeth", "A4. brushing hair", "A5. drop", "A6. pickup",
    "A7. throw", "A8. sitting down", "A9. standing up (from sitting position)", "A10. clapping", "A11. reading",
    "A12. writing", "A13. tear up paper", "A14. wear jacket", "A15. take off jacket", "A16. wear a shoe",
    "A17. take off a shoe", "A18. wear on glasses", "A19. take off glasses", "A20. put on a hat",
    "A21. take off a hat", "A22. cheer up", "A23. hand waving", "A24. kicking something", "A25. reach into pocket",
    "A26. hopping (one foot jumping)", "A27. jump up", "A28. make a phone call", "A29. playing with phone",
    "A30. typing on a keyboard", "A31. pointing to something with finger", "A32. taking a selfie", "A33. check time (from watch)",
    "A34. rub two hands together", "A35. nod head", "A36. shake head", "A37. wipe face", "A38. salute", "A39. put the palms together",
    "A40. cross hands in front (say stop)", "A41. sneeze", "A42. staggering", "A43. falling", "A44. touch head (headache)",
    "A45. touch chest (stomachache)", "A46. touch back (backache)", "A47. touch neck (neckache)", "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)", "A50. punching other person", "A51. kicking other person",
    "A52. pushing other person", "A53. pat on back of other person", "A54. point finger at the other person",
    "A55. hugging other person", "A56. giving something to other person", "A57. touch other person's pocket",
    "A58. handshaking", "A59. walking towards each other", "A60. walking apart from each other", "A61. put on headphone",
    "A62. take off headphone", "A63. shoot at the basket", "A64. bounce ball", "A65. tennis bat swing", "A66. juggling table tennis balls",
    "A67. hush (quite)", "A68. flick hair", "A69. thumb up", "A70. thumb down", "A71. make ok sign", "A72. make victory sign",
    "A73. staple book", "A74. counting money", "A75. cutting nails", "A76. cutting paper (using scissors)", "A77. snapping fingers",
    "A78. open bottle", "A79. sniff (smell)", "A80. squat down", "A81. toss a coin", "A82. fold paper", "A83. ball up paper",
    "A84. play magic cube", "A85. apply cream on face", "A86. apply cream on hand back", "A87. put on bag", "A88. take off bag",
    "A89. put something into a bag", "A90. take something out of a bag", "A91. open a box", "A92. move heavy objects", "A93. shake fist",
    "A94. throw up cap", "A95. hands up (both hands)", "A96. cross arms", "A97. arm circles", "A98. arm swings", "A99. running on the spot",
    "A100. butt kicks (kick backward)", "A101. cross toe touch", "A102. side kick", "A103. yawn", "A104. stretch oneself",
    "A105. blow nose", "A106. hit other person with something", "A107. wield knife towards other person",
    "A108. knock over other person (hit with body)", "A109. grab other person’s stuff", "A110. shoot at other person with a gun",
    "A111. step on foot", "A112. high-five", "A113. cheers and drink", "A114. carry something with other person",
    "A115. take a photo of other person", "A116. follow other person", "A117. whisper in other person’s ear",
    "A118. exchange things with other person", "A119. support somebody with hand", "A120. finger-guessing game (playing rock-paper-scissors)"
]


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, random_scale=False, random_mask=False, window_size=-1, 
                 normalization=False, debug=False, use_mmap=False, bone=False, vel=False, random_miss=False, 
                 random_miss_type=None, miss_amount=0.0, structured_miss=False, structured_miss_type=None,
                 structured_res=1, chunks=None, FPS=30, mitigation=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param random_scale: scale skeleton length
        :param random_mask: mask some frames with zero
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.random_scale = random_scale
        self.random_mask = random_mask
        self.bone = bone
        self.vel = vel
        self.random_miss = random_miss
        self.random_miss_type = random_miss_type
        self.miss_amount = miss_amount
        self.structured_miss = structured_miss
        self.structured_miss_type = structured_miss_type
        self.structured_res = structured_res
        self.chunks = chunks
        self.FPS = FPS
        self.mitigation = mitigation
        
        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C T V M   (output)
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM

        if index == 0:
            print('before deg sample len:', valid_frame_num)

        if self.random_miss and self.structured_miss:
            sys.exit('Both random_miss and structured_miss cannot be true at the same time')

        if self.random_miss:
            if index==0:
                print('Dropping frames randomly')
                print(f"Method for dropping frames: {self.random_miss_type}")

                if self.miss_amount > 1:
                    sys.exit('Please set the miss_amount to a value less than or equal to 1')
                print(f"amount of frames to drop: {self.miss_amount*100}%")

            data_numpy = self.apply_degradation(data_numpy, valid_frame_num, index)

        elif self.structured_miss:
            if index==0:
                print(f'Structured dropout, using method {self.structured_miss_type}')
                if self.structured_miss_type == 'frame_rate':  
                    print(f'Old FPS: 30, New FPS: {self.FPS}')
                    print(f'Number of chunks {self.chunks}')
                elif self.structured_miss_type == 'reduced_resolution':
                    print(f'Structured drop frequency is {self.structured_res}')
                else:
                    sys.exit('self.structured_miss_type has been set incorrectly, please enter either "frame_rate" or "reduced_resolution"')

            data_numpy = self.apply_structured_degradation(data_numpy, valid_frame_num, index)
            
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        if index == 0:
            print('after deg sample len:', valid_frame_num)

        
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.random_scale:
            data_numpy = tools.random_scale(data_numpy)
        if self.random_mask:
            data_numpy = tools.random_mask(data_numpy)
        # if self.bone:
        #     from .bone_pairs import ntu_pairs
        #     bone_data_numpy = np.zeros_like(data_numpy)
        #     for v1, v2 in ntu_pairs:
        #         bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        #     data_numpy = bone_data_numpy
        # if self.vel:
        #     data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        #     data_numpy[:, -1] = 0

        return data_numpy, label, index

    def apply_degradation(self, data, no_of_frames, index):

        if self.random_miss_type is None:
            sys.exit('random_miss_type is not set')
        
        if index==0:
            print(f"random_miss_type is {self.random_miss_type}")

        if self.random_miss_type == 'delete':
            data = self.delete_frames(data, no_of_frames, index)
        elif self.random_miss_type == 'drop_next':
            data = self.drop_next_frames(data, no_of_frames, index)
        elif self.random_miss_type == 'drop_previous':
            data = self.drop_prev_frames(data, no_of_frames, index)
        elif self.random_miss_type == 'interpolate':
            data = self.dropout_interpolate(data, no_of_frames, index)
        else:
            sys.exit('Miss type has not been implemented, check for spelling errors.')

        return data

    def delete_frames(self, data, no_of_frames, index):
        
        channels, origin_t, num_joint, num_people = np.shape(data) #input shape: C, T, V, M
        if index == 0:
            print('Amount of frames to drop:', self.miss_amount*no_of_frames)
            print('Length of stream:',  no_of_frames)

        t_index = np.arange(no_of_frames)
        frames_to_drop = int(self.miss_amount*no_of_frames)
        filter = np.random.choice(t_index,size=frames_to_drop, replace=False) 
        indices = np.argwhere(np.isin(t_index,filter))
        valid_t_index = np.delete(t_index,indices)
        arr = data[:,valid_t_index,:,:]
        output = np.zeros((channels, origin_t, num_joint, num_people), dtype=np.float32)
        new_size = no_of_frames - frames_to_drop
        output[:,:new_size,:,:] = arr

        if index == 0:
            print('output:', output.shape)
            print('new size:', new_size)

        return output

    def drop_next_frames(self, data, no_of_frames, index):
        
        channels, origin_t, num_joint, num_people = np.shape(data)
        t_index = np.arange(no_of_frames)
        filter = np.random.choice(t_index,size=int((self.miss_amount*no_of_frames)), replace=False)
        indices = np.argwhere(np.isin(t_index,filter))

        indices = np.sort(indices) 
        indices = indices[::-1] 

        for j in range(len(indices)):
            if indices[j] != no_of_frames-1:
                data[:, indices[j],:,:] = data[:, indices[j]+1,:,:] 
            else:
                continue

        #data = self.interpolate(data, num_joint, origin_t, channels, num_people)

        return data
    
    def drop_prev_frames(self, data, no_of_frames, index):
        pass

    def dropout_interpolate(self, data, no_of_frames, index):

        channels, origin_t, num_joint, num_people = np.shape(data) #input shape: C, T, V, M
        t_index = np.arange(no_of_frames)
        t_index = t_index[1:-1] # Remove first and last frames so these can't be interpolated or accidentally get wrapped. 
        drop_amount = int(self.miss_amount*no_of_frames)
        if drop_amount > len(t_index):
            drop_amount = len(t_index)
        
        filter = np.random.choice(t_index,size=drop_amount, replace=False)
        indices = np.argwhere(np.isin(t_index,filter))
        indices = np.sort(indices) + 1

        sub_arrays = np.split(indices, np.flatnonzero(np.diff(indices.T)!=1) + 1)
        for i in range(len(sub_arrays)):
            sub_arrays[i] = sub_arrays[i].flatten()

        for sub_array in sub_arrays:
            len_sub_array = len(sub_array)
            if len_sub_array > 1:
                for j in range(len_sub_array): 
                    if sub_array[len_sub_array-1]+1 == origin_t:
                        end = data[:,sub_array[len_sub_array-1],:,:]
                    else:
                        end = data[:,sub_array[len_sub_array-1]+1,:,:]
                    start = data[:,sub_array[0]-1,:,:]
                    
                    inc = (end-start) / (len_sub_array+1)
                    data[:,sub_array[j],:,:] = inc * (j+1) + start
            else:
                if len(sub_array) == 0:
                    continue

                if sub_array[0] == 299:
                    data[:,sub_array[0],:,:] = data[:,sub_array[0],:,:]
                else:
                    data[:,sub_array[0]] = (data[:,sub_array[0]-1,:,:] + data[:,sub_array[0]+1,:,:]) / 2

        return data
    
    def interpolate(self, data, num_joint, origin_t, channels, num_people, valid_t_index=None):

        if valid_t_index is not None:
            data = data[:,:,valid_t_index,:]
        
        data = data.transpose(3,1,2,0)  # M,valid_T,V,C
        data_rescaled = np.zeros((num_people, origin_t, num_joint, channels))
        for i in range(num_people):
            data_rescaled[i] = cv2.resize(data[i], (num_joint, origin_t), interpolation=cv2.INTER_LINEAR)
        data = data_rescaled.transpose(3,1,2,0) # back to C,T,V,M

        return data

    def apply_structured_degradation(self, data, no_of_frames, index):

        if self.structured_miss_type == 'reduced_resolution':
            data = self.reduced_resolution(data, no_of_frames, index)
        elif self.structured_miss_type == 'frame_rate':
            data = self.frame_rate(data, no_of_frames, index)
        else:
            sys.exit('self.structured_miss_type not selected correctly')
        
        return data

    def reduced_resolution(self, data, no_of_frames, index):

        if self.mitigation:
            channels, origin_t, num_joint, num_people = np.shape(data) #input shape: C, T, V, M   
            output = np.zeros((channels, origin_t, num_joint, num_people), dtype=np.float32)
            output[:,:no_of_frames:self.structured_res,:,:] = data[:,:no_of_frames:self.structured_res,:,:]

            x = np.arange(no_of_frames)
            y = x[:no_of_frames:self.structured_res]

            selected_indices = np.where(np.isin(x, y))[0]
            all_indices = np.arange(len(x))
            unselected_indices = np.setdiff1d(all_indices, selected_indices)

            sub_arrays = np.split(unselected_indices, np.flatnonzero(np.diff(unselected_indices.T)!=1) + 1)

            for sub_array in sub_arrays:
                if len(x)-1 in sub_array:
                    continue
                len_sub_array = len(sub_array)
                if len_sub_array > 1:
                    for i in range(len_sub_array):
                        
                        if sub_array[len_sub_array-1]+1 == len(x):
                            end = data[:,sub_array[len_sub_array-1],:,:]
                        else:
                            end = data[:,sub_array[len_sub_array-1]+1,:,:]
                        start = data[:,sub_array[0]-1,:,:]
                        
                        inc = (end-start) / (len_sub_array+1)
                        output[:,sub_array[i],:,:] = inc * (i+1) + start
                else:
                    if len(sub_array) == 0:
                        continue
                    output[:,sub_array[0]] = (data[:,sub_array[0]-1,:,:] + data[:,sub_array[0]+1,:,:]) / 2
        
        else:   
            channels, origin_t, num_joint, num_people = np.shape(data) #input shape: C, T, V, M        
            arr = data[:,:no_of_frames:self.structured_res,:,:]
            output = np.zeros((channels, origin_t, num_joint, num_people), dtype=np.float32)
            output[:,:arr.shape[1],:,:] = arr
    
            if index == 0:
                print(no_of_frames, arr.shape)
                #np.savetxt('./work_dir/joints/testing/reduced_resolution_stride_'+ str(self.structured_res) +'.txt', arr[0,:,0,0])
                #np.savetxt('./work_dir/joints/testing/reduced_resolution_truth_'+ str(self.structured_res) + '.txt', output[0,:,0,0])
                #np.savetxt('./work_dir/joints/testing/reduced_resolution_baseline.txt', data_numpy[0,:,0,0])

        return output
        
    def frame_rate(self, data, no_of_frames, index):

        channels, origin_t, num_joint, num_people = np.shape(data) #input shape: C, T, V, M
        FPS_Drop = self.FPS/30

        if self.chunks == 1: # single chunk
            if self.FPS == 30:
                sys.exit('FPS is already 30, no need to reduce frame rate')
            chunk_length = int(no_of_frames - no_of_frames*FPS_Drop)
            if chunk_length >= no_of_frames - 1:
                chunk_length = no_of_frames - 2
            max_chunk_start = no_of_frames - chunk_length
            if max_chunk_start-1 <= 0:
                print(index, no_of_frames, chunk_length, max_chunk_start) # 36143 10 9 1
            chunk_start = np.random.randint(0, high=max_chunk_start-1)
            list_of_indices = np.arange(chunk_start, chunk_start+chunk_length)
            arr = np.delete(data, list_of_indices, axis=1)
            new_length = int(no_of_frames - chunk_length)
            
            output = np.zeros((channels, origin_t, num_joint, num_people), dtype=np.float32)
            output[:,:new_length,:,:] = arr[:,:new_length,:,:]

            if index == 0:
                print('Single chunk:', no_of_frames, new_length)
                print('Chunk start:', chunk_start)
                print('Chunk length:', chunk_length)

            if self.mitigation:
                output = data.copy()
                start = data[:,chunk_start,:,:]
                end = data[:,chunk_start + chunk_length,:,:]
                
                for i in range(0, chunk_length):
                    increment = (end-start)/chunk_length
                    output[:,chunk_start+i,:,:] = start + increment*i

                if index == 0:
                    print('Mitigating frames: ', data.shape[1])
        
        else:
            print('Multi Chunking not implemented yet')

        return output
    
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
