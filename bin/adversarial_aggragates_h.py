import dataviz.distplot as distplot
import bin.paths as paths

if __name__ == '__main__':
    sort_dict = "HeuristicAntDictAntRecFUTURE"
    param_num = 3
    file_num = 2
    formt = 'svg'
    variable = 'action'
    # offset = -1000
    data = distplot.file2dataframe(paths.root_dir, variable, 1000, paths.tested_params, size=9999)
    data1 = data.dropna()
    if variable is 'action':
        import numpy as np
        import itertools
        data2 = data1.copy()
        for key, value in data1.items():
            for ind, el in enumerate(value):
                data2[key][ind] = np.squeeze(el, axis=1)
                data2[key][ind] = np.array(list(itertools.compress(el, [i not in [0] for i in el])))
        data1 = data2
    # import pandas as pd

    # df = pd.DataFrame(columns=data1.columns, index=data1.index)
    # for dkey in data1.keys():
    #     for i in range(data1.index.size):
    #         df[dkey][i] = data1[dkey][i][(data1[dkey][i] != 0.)[:, 0]]

    # data['f10f10b99b03k0001k0001m10'] = data['f10f10b99b03k0001k0001m10'].drop([9])
    data1.rename(columns={'f10f10b03b03k001k001m10': 'Setup1', 'f10f10b099b03k001k001m10': 'Setup2',
                          'f10f10b099b099k001k001m10': 'Setup3'}, inplace=True)

    # data.to_csv("../data/adversarial.csv")

    means, errors = distplot.calculate_avg_behavior(data1)
    distplot.plotly_box_plot(means, paths.save_path,
                             name='boxplot_nozero_{}_avg_behavior'.format(variable), fmt=formt)
