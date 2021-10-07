import argparse


def parse_cmd():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split_times", type=int,default = 20)
    parser.add_argument("--method", type=str, default="LR")
    parser.add_argument("--use_value", type=str, default=['f1_score','f1_score_var'],help = 'f1_score/accuracy/precision/recall + _var', nargs='+')
    parser.add_argument("--use_num", type=int, default=[85],help='5 * x for selecting the number for use', nargs='+')
    parser.add_argument("--target_value", type=str, default='f1_score', help = 'f1_score/accuracy/roc_auc')
    parser.add_argument("--train_alg", type=str, default = 'RF_score', help  = 'RF_score/LR_score')
    parser.add_argument("--meta_value", type=str, default=[],help = 'feature_cnt/max_coev/label_ratio', nargs='+')
    parser.add_argument("--pilot_length", type=int, default = 100)
    parser.add_argument("--class_standard", type=int, default=[128,250,440,800])
    parser.add_argument("--print_coef", type=bool, default=False)
    parser.add_argument("--use_auto_ml", type=bool, default=False)
    parser.add_argument("--use_file", type=str, default=['dataset_data_100.csv', 'dataset_data_noised.csv'], nargs='+')
    args = parser.parse_args()
    return args
