from feature_engineering.feature_generate_memory import *
from feature_engineering.utils_memory import categories_to_int
from feature_engineering.utils_memory import combine_feature_tuples
from feature_engineering.memory_no_encode import Memory


class Pipline(object):
    def __init__(self, args):
        self.ori_dataframe = args['dataframe']  # 这个属性保存原始的dataframe，所有step不再修改
        self.continuous_columns = args['continuous_columns']
        self.discrete_columns = args['discrete_columns']
        self.mode = args['mode']
        self.isvalid = args['isvalid']
        self.memory = args['memory']
        if self.mode == 'classify':
            self.label = categories_to_int(self.ori_dataframe[args['label_name']].values.reshape(-1),
                                           -1, self.memory, self.isvalid)
            self.ori_dataframe = self.ori_dataframe.copy()
            self.ori_dataframe[args['label_name']] = self.label
        else:
            self.label = self.ori_dataframe[args['label_name']].values
        self.refresh_states()

    @property
    def get_memory(self):
        return self.memory

    def refresh_states(self):
        '''重置随着episode变化的属性'''
        # 随着step变化的属性，每个episode需要被重置
        self.discrete_fe_2_int_type()
        self.ori_cols_continuous = self.ori_dataframe[self.continuous_columns].values
        self.ori_c_columns_norm = self.continous_fe_2_norm()
        self.discrete_fes_out_after_encode = None
        self.c_fes_norm_out = None
        self.c_fes_scale_out = None

    def __refresh_continuous_actions__(self, actions_c):
        '''每个step更新连续特征操作'''
        # print(actions_c)
        self.value_convert = actions_c['value_convert'] if 'value_convert' in actions_c else {}  # dict
        self.add_ = actions_c['add'] if 'add' in actions_c else []
        self.subtract_ = actions_c['subtract'] if 'subtract' in actions_c else []
        self.multiply_ = actions_c['multiply'] if 'multiply' in actions_c else []
        self.divide_ = actions_c['divide'] if 'divide' in actions_c else []
        self.bins = actions_c['bins'] if 'bins' in actions_c else {}  # 只对连续特征分箱

    def __refresh_discrete_actions__(self, actions_d):
        '''每个mdp step需运行该函数'''
        self.Cn2 = actions_d['two'] if 'two' in actions_d else []
        self.Cn3 = actions_d['three'] if 'three' in actions_d else []
        self.Cn4 = actions_d['four'] if 'four' in actions_d else []
        self.bins = actions_d['bins1'] if 'bins1' in actions_d else {}

    def discrete_fe_2_int_type(self):
        all_names = self.ori_dataframe.columns.tolist()[:-1]
        for discrete_col_name in self.discrete_columns:
            col_index = all_names.index(discrete_col_name)
            ori_col = self.ori_dataframe[discrete_col_name].values
            int_type_col = categories_to_int(ori_col, col_index, self.memory, self.isvalid)
            self.ori_dataframe = self.ori_dataframe.reset_index(drop=True)
            self.ori_dataframe[discrete_col_name] = int_type_col
        ori_dataframe_discrete = self.ori_dataframe[self.discrete_columns].copy()
        self.ori_dataframe.astype('float32')
        ori_dataframe_discrete.astype('float32')
        self.ori_cols_discrete = ori_dataframe_discrete.values
        self.ori_cols = self.ori_dataframe.values[:, :-1]

    def continous_fe_2_norm(self):
        '''normalization for continuous features'''
        fes_after_norm = None
        for col_index in range(len(self.continuous_columns)):
            ori_col = self.ori_cols_continuous[:, col_index]
            scaled_fe = normalization(ori_col,col_index,
                                      self.memory,self.isvalid)
            if isinstance(fes_after_norm, np.ndarray):
                fes_after_norm = np.hstack((fes_after_norm, scaled_fe))
            else:
                fes_after_norm = scaled_fe
        return fes_after_norm

    def single_fe_operations(self, ori_fes):
        '''convert operations for continuous features'''
        for index, operation in self.value_convert.items():
            ori_col = self.ori_cols_continuous[:, index]
            if operation != 'None':
                new_fe = globals()[operation](ori_col)
                new_fe_index = self.ori_cols_continuous.shape[1]
                self.ori_cols_continuous = np.hstack((self.ori_cols_continuous, new_fe))
                new_fe = new_fe.reshape(-1)
                if len(new_fe) > 0:
                    fe_norm = normalization(new_fe,new_fe_index,self.memory,self.isvalid)
                    if isinstance(ori_fes, np.ndarray):
                        ori_fes = np.hstack((ori_fes, fe_norm))
                    else:
                        ori_fes = fe_norm
        return ori_fes

    def arithmetic_operations(self, ori_fes):
        '''add/sub/multi/divide operations for continuous features'''
        operations = ['add', 'subtract', 'multiply', 'divide']
        feature_informations = [self.add_, self.subtract_, self.multiply_, self.divide_]
        for i, feature_information in enumerate(feature_informations):
            combine_feature_tuples_list = combine_feature_tuples(feature_information, 2)
            operation = operations[i]
            for col_index_tuple in combine_feature_tuples_list:
                col1_index, col2_index = col_index_tuple
                col1 = self.ori_cols_continuous[:, col1_index]
                col2 = self.ori_cols_continuous[:, col2_index]
                if operation != 'None':
                    new_fe = globals()[operation](col1, col2)
                    ori_len = self.ori_cols_continuous.shape[1]
                    self.ori_cols_continuous = np.hstack((self.ori_cols_continuous,new_fe))
                    for i in range(new_fe.shape[1]):
                        new_fe_index = ori_len + i
                        fe_norm = normalization(new_fe[:, i],new_fe_index,self.memory,self.isvalid)
                        if isinstance(ori_fes, np.ndarray):
                            ori_fes = np.hstack((ori_fes, fe_norm))
                        else:
                            ori_fes = fe_norm
        return ori_fes

    def binning_operation(self):
        '''bin step for all columns,'''
        all_names = self.ori_dataframe.columns.tolist()[:-1]
        ori_cols = self.ori_cols.copy()
        for index, bins in self.bins.items():
            col_name = all_names[index]
            bins = int(bins)
            ori_fe = ori_cols[:, index]
            if len(np.unique(ori_fe)) > bins or self.isvalid:
                if col_name in self.discrete_columns:
                    ori_fe = binning_for_discrete(ori_fe, bins, self.mode, self.label,
                                                  index, self.memory, self.isvalid)
                else:
                    ori_fe, fre_list = binning(ori_fe, bins, index,
                                               memory = self.memory, isvalid = self.isvalid)
                    ori_fe = ori_fe.reshape(-1)
                ori_cols[:, index] = ori_fe
        self.ori_cols = ori_cols

    # def select_features(self):
    #     '''select features due to RL agent'''
    #     ori_cols = self.ori_cols.copy()
    #     ori_mask = np.ones(ori_cols.shape[1])
    #     for index,mask in self.selector.items():
    #         ori_mask[index] = int(mask)
    #     selected_index = np.argwhere(ori_mask == 1).reshape(-1)
    #     ori_cols = ori_cols[:,selected_index]
    #     self.ori_cols = ori_cols

    def feature_cross_operations(self, ori_fes = None):
        '''feature cross combine operation for discrete features'''
        operations = ['Cn2','Cn3','Cn4']
        feature_informations = [self.Cn2,self.Cn3,self.Cn4]
        for i, feature_information in enumerate(feature_informations):
            operation = operations[i]
            if operation != 'None':
                type = int(operation[-1])
                combine_feature_tuples_list = combine_feature_tuples(feature_information, type)
                for combine_feature_tuple in combine_feature_tuples_list:
                    combine_feature_index_list = list(combine_feature_tuple)
                    ori_cols = self.ori_cols[:, combine_feature_index_list]
                    # print(combine_feature_index_list)
                    new_fe = features_combine_ori(ori_cols, combine_feature_index_list,
                                              self.memory, self.isvalid)
                    if isinstance(ori_fes, np.ndarray):
                        ori_fes = np.hstack((ori_fes, new_fe))
                    else:
                        ori_fes = new_fe
        if isinstance(ori_fes,np.ndarray):
            self.ori_cols = np.hstack((self.ori_cols,ori_fes))

    def process_discrete(self, actions):
        '''处理所有离散变量'''
        self.__refresh_discrete_actions__(actions)
        # 四步，分箱和combine
        self.binning_operation() # 第一次分箱
        self.feature_cross_operations() # feature_combine
        return self.ori_cols, self.label

    def process_continuous(self, actions):
        '''处理所有连续变量'''
        self.__refresh_continuous_actions__(actions)
        new_fes = None
        new_fes = self.single_fe_operations(new_fes)
        new_fes = self.arithmetic_operations(new_fes)
        if not isinstance(self.c_fes_norm_out, np.ndarray):
            if isinstance(new_fes, np.ndarray):
                self.c_fes_norm_out = \
                    np.hstack((self.ori_c_columns_norm, new_fes))
            else:
                self.c_fes_norm_out = self.ori_c_columns_norm
        else:
            if isinstance(new_fes, np.ndarray):
                self.c_fes_norm_out = \
                    np.hstack((self.c_fes_norm_out, new_fes))
        out_normed = self.c_fes_norm_out
        return out_normed, self.label


def generate_fes(actions_c, actions_d, args: dict) -> tuple:
    pipline = Pipline(args)
    c_fes, d_fes = [], []
    if actions_d != [{}]:
        for action_d in actions_d:
            d_fes, _ = pipline.process_discrete(action_d)
    if actions_c != [{}]:
        for action_c in actions_c:
            c_fes, _ = pipline.process_continuous(action_c)
    return c_fes, d_fes, pipline.label


if __name__ == "__main__":
    import pandas as pd

    file_name = r'F:\works\2021\AutoML\AutoML\Plan_C_CD\data\adult.csv'
    c_columns = ['age', 'fnlwgt', 'earnings', 'loss', 'hour', 'edu_nums']
    d_columns = ['work_cate', 'education', 'marital', 'profession', 'relation', 'race', 'gender', 'motherland']
    target = 'label'
    dataframe = pd.read_csv(file_name)
    d0 = dataframe.iloc[:4000, :]
    d0 = dataframe.iloc[:4000, :]
    c_op = [{'value_convert': {0: 'log', 1: 'None', 2: 'sqrt', 3: 'log', 4: 'None', 5: 'log'}},
            {'add': [9], 'subtract': [1, 3, 4, 7], 'multiply': [2, 5, 6, 8], 'divide': [0]}]
    d_op = [{'bins': {0: '5', 1: '5', 2: '10', 3: '10', 4: '8', 5: '8', 6: '8', 7: '10', 8: '5', 9: '10', 10: '10',
                      11: '8', 12: '8', 13: '10'}}, {'two': [2, 3, 4, 8, 10], 'three': [0], 'four': []}]
    # d_op = [{'bins': {0: '5', 1: '5', 2: '10', 3: '10', 4: '8', 5: '8', 6: '8', 7: '10', 8: '5', 9: '10', 10: '10',
    #                   11: '8', 12: '8', 13: '10'}}]
    memory = Memory()
    pipline_args = {'dataframe': d0,
                    'continuous_columns': c_columns,
                    'discrete_columns': d_columns,
                    'label_name': target,
                    'mode': 'classify',
                    'isvalid': False,
                    'memory': memory}

    pipline_args1 = {'dataframe': d0,
                     'continuous_columns': c_columns,
                     'discrete_columns': d_columns,
                     'label_name': target,
                     'mode': 'classify',
                     'isvalid': True,
                     'memory': memory}

    c_fes, d_fes, fes_gen_all = generate_fes(c_op, d_op, pipline_args)
    # print('---------------')
    print(c_fes.shape)
    print(d_fes.shape)

    # exit()

    c_fes1, d_fes1, fes_gen_all1 = generate_fes(c_op, d_op, pipline_args1)
    print(c_fes1.shape)
    print(d_fes1.shape)

