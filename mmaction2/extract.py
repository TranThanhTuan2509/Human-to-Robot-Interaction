# path = '/home/tuan/Downloads/model_name.txt'
#
# with open(path, 'r') as file:
# 	output = [model[1:-2] + '\n' for model in file.read().split(",") if 'sthv2' in model]
# 	print(output)
#
# with open('/home/tuan/Downloads/model.txt', 'w') as file:
# 	out = file.writelines(output)

path = '/home/tuan/Documents/Code/mmaction2/tools/data/sthv2/label_map.txt'
with open(path, 'r') as file:
    lines = file.readlines()
    lines = {i:x.strip() for i,x in enumerate(lines)}

print(lines)
