function to_file(data)
    for i = 1:2858
        x = cumtrapz(data{i}(1, :));
        y = cumtrapz(data{i}(2, :));
        f = data{i}(3, :);
        mat = [x' y' f'];
        filename = strcat('letter_', sprintf('%04d',i), '.csv');
        csvwrite(filename, mat)
    end