import numpy as np

def load_data(filename, leading_row=1, leading_column=1, delimiter=','):
    with open(filename, 'rb') as f:
        input_matrix = f.read().splitlines()
        output_matrix = []
        for i, row in enumerate(input_matrix):
            if i <= leading_row:
                continue
            row_data = row.split(delimiter)
            row_append = []
            for j, ele in enumerate(row_data):
                if j <= leading_column:
                    continue
                row_append.append(int(ele.replace('"', '')))
            output_matrix.append(row_append)
    print output_matrix[0]
    return np.array(output_matrix, dtype=np.float32)
