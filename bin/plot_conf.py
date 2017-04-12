import dataviz.distplot as distplot
import bin.paths as paths

if __name__ == "__main__":
    idx = [i for i in range(0, 11)]
    data = distplot.file2dataframe(paths.root_dir, "actions", 1000)
    for el in paths.prefixes:
        if el == 2:
            continue
        distplot.hist_by_tag(data, ''.join(['f', str(el)]), idx, True, paths.bins, paths.save_path)
