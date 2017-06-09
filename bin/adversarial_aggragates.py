import dataviz.distplot as distplot
import bin.paths as paths

if __name__ == '__main__':
    sort_dict = "AntDictAntRec"
    param_num = 3
    file_num = 2
    formt = 'svg'
    variable = 'action'
    offset = -50
    data = distplot.file2dataframe(paths.root_dir, variable, 1000, paths.tested_params, size=9999)

    means, errors = distplot.calculate_avg_behavior(data)
    distplot.plotly_box_plot(means, paths.save_path,
                             name='boxplot2_{}_avg_behavior'.format(variable), fmt=formt)
