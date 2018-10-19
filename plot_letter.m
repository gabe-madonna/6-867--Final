function plot_letter(data, index)
    x = cumtrapz(data{index}(1, :));
    y = cumtrapz(data{index}(2, :));
    plot(x, y)
