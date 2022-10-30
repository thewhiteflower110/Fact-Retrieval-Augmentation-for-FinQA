#evaluating Program Generator's effictiency from program and result of prog
max_seq_length = 512
max_program_length = 30
#used to convert text processed rows to numerics
def str_to_num(text):
    text = text.replace("$","")
    text = text.replace(",", "")
    text = text.replace("-", "")
    text = text.replace("%", "")
    try:
        num = float(text)
    except ValueError:
        if "const_" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


#Produces predicted recursive program to list program
#list program is helpful to evaluate
def reprog_to_seq(prog_in, is_gold):
    '''
    predicted recursive program to list program
    ["divide(", "72", "multiply(", "6", "210", ")", ")"]
    ["multiply(", "6", "210", ")", "divide(", "72", "#0", ")"]
    '''

    st = []
    res = []

    try:
        num = 0
        for tok in prog_in:
            if tok != ")":
                st.append(tok)
            else:
                this_step_vec = [")"]
                for _ in range(3):
                    this_step_vec.append(st[-1])
                    st = st[:-1]
                res.extend(this_step_vec[::-1])
                st.append("#" + str(num))
                num += 1
    except:
        if is_gold:
            raise ValueError

    return res

#calculate the numerical results of the program
def eval_program(program):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"

    try:
        program = program[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program = "|".join(program)
        steps = program.split(")")[:-1]

        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if "#" in arg1:
                arg1 = res_dict[int(arg1.replace("#", ""))]
            else:
                arg1 = str_to_num(arg1)
                if arg1 == "n/a":
                    invalid_flag = 1
                    break

            if "#" in arg2:
                arg2 = res_dict[int(arg2.replace("#", ""))]
            else:
                arg2 = str_to_num(arg2)
                if arg2 == "n/a":
                    invalid_flag = 1
                    break

            if op == "add":
                this_res = arg1 + arg2
            elif op == "subtract":
                this_res = arg1 - arg2
            elif op == "multiply":
                this_res = arg1 * arg2
            elif op == "divide":
                this_res = arg1 / arg2
            elif op == "exp":
                this_res = arg1 ** arg2

            res_dict[ind] = this_res

        if this_res != "n/a":
            this_res = round(this_res, 5)

    except:
        invalid_flag = 1

    return invalid_flag, this_res

#evaluates program and result of program against true results
def evaluate_result(all_nbest, json_ori, program_mode):
    '''
    execution acc
    program acc
    '''

    data = all_nbest

    with open(json_ori) as f_in:
        data_ori = json.load(f_in)

    data_dict = {}
    for each_data in data_ori:
        assert each_data["uid"] not in data_dict
        data_dict[each_data["uid"]] = each_data

    exe_correct = 0

    res_list = []
    all_res_list = []

    for tmp in data:
        each_data = data[tmp][0]
        each_id = each_data["id"]

        each_ori_data = data_dict[each_id]
        gold_res = each_ori_data["qa"]["answer"]

        pred = each_data["pred_prog"]
        gold = each_data["ref_prog"]

        if program_mode == "nest":
            if pred[-1] == "EOF":
                pred = pred[:-1]
            pred = reprog_to_seq(pred, is_gold=False)
            pred += ["EOF"]
            gold = gold[:-1]
            gold = reprog_to_seq(gold, is_gold=True)
            gold += ["EOF"]

        invalid_flag, exe_res = eval_program(pred)

        if invalid_flag == 0:
            if exe_res == gold_res:
                exe_correct += 1

        each_ori_data["qa"]["predicted"] = pred

        if exe_res != gold_res:
            res_list.append(each_ori_data)
        all_res_list.append(each_ori_data)

    exe_acc = float(exe_correct) / len(data)

    print("All: ", len(data))
    print("Exe acc: ", exe_acc)

    return exe_acc