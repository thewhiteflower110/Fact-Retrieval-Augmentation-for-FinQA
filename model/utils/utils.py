def read_txt(input_path):
    """Read a txt file into a list."""
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items

def get_op_const_list():
    op_list_file = "../txt_files/operation_list.txt"
    const_list_file = "../txt_files/constant_list.txt"
    op_list = read_txt(op_list_file)
    op_list = [op + '(' for op in op_list]
    op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
    const_list = read_txt(const_list_file)
    const_list = [const.lower().replace('.', '_') for const in const_list]
    return op_list, const_list
