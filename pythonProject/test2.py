import xml.etree.ElementTree as ET

# 解析TMX文件
tree = ET.parse('aaa.tmx')
root = tree.getroot()

# 打开文件用于写入
with open('cn.txt', 'w', encoding='utf-8') as cn_file, open('en.txt', 'w', encoding='utf-8') as en_file:
    # 遍历所有的tuv元素
    for tuv in root.iter('tuv'):
        lang = tuv.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
        seg = tuv.find('seg').text
        if lang == 'zh-CN':
            cn_file.write(seg + '\n')
        elif lang == 'en-US':
            en_file.write(seg + '\n')

print("success")
