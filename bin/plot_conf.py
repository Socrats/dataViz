import dataviz.distplot as distplot
import bin.paths as paths

# data_dict = {}
# for index, payoff_file in zip(paths.prefixes, paths.payoff_files):
#     actions_vector = np.load(payoff_file)
#     actions_vector = actions_vector[0][-1000:]
#     data_dict[''.join(['f', str(index)])] = actions_vector
#
# actions_df = pd.DataFrame(data_dict)
# actions_df = actions_df[['f1', 'f2', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f20']]
# # print(actions_df)
#
# actions_df.plot.hist(stacked=True, bins=paths.bins)
# actions_df.diff().hist(color='k', alpha=0.5, bins=paths.bins)
# plt.axhline(0, color='k')
# plt.show()

# actions_files = utils.get_files(paths.root_dir, "actions")
# data = pd.DataFrame(np.load(actions_files['f9'][1])[-1000:].transpose())
# bootstrap_plot(data, size=50, samples=500, color='grey')
# plt.show()


if __name__ == "__main__":
    idx = [i for i in range(0, 11)]
    sort_dict = {'f1': 0, 'f2': 1, 'f5': 3, 'f6': 4, 'f7': 5, 'f8': 6, 'f9': 7, 'f10': 8, 'f11': 9, 'f12': 10,
                 'f13': 11, 'f14': 12, 'f15': 13, 'f20': 14}
    data = distplot.file2dataframe(paths.root_dir, "actions", 1000, sort_dict)
    for el in paths.prefixes:
        if el == 2:
            continue
        distplot.hist_by_tag(data, ''.join(['f', str(el)]), idx, paths.bins, paths.save_path, density=True)
