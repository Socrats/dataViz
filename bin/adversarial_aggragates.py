import dataviz.distplot as distplot
import bin.paths as paths

if __name__ == '__main__':
    sort_dict = "AntDictAntRec"
    param_num = 3
    file_num = 2
    formt = 'svg'
    variable = 'acceptance'
    # offset = -1000
    data = distplot.file2dataframe(paths.root_dir, variable, 1000, paths.tested_params, size=9999)
    data1 = data.dropna()

    # import pandas as pd

    # df = pd.DataFrame(columns=data1.columns, index=data1.index)
    # for dkey in data1.keys():
    #     for i in range(data1.index.size):
    #         df[dkey][i] = data1[dkey][i][(data1[dkey][i] != 0.)[:, 0]]

    data['f10f10b99b03k0001k0001m10'] = data['f10f10b99b03k0001k0001m10'].drop([9])
    data.rename(columns={'f10f10b03b03k0001k0001m10': 'Setup1', 'f10f10b03b03k0001k1m10': 'Setup2',
                         'f10f10b99b03k0001k0001m10': 'Setup3', 'f10f10b99b03k0001k1m10': 'Setup4'}, inplace=True)

    # data.to_csv("../data/adversarial.csv")

    means, errors = distplot.calculate_avg_behavior(data)
    distplot.plotly_box_plot(means, paths.save_path,
                             name='boxplot_nozero_{}_avg_behavior'.format(variable), fmt=formt)
