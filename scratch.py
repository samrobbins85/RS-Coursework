from prompt_toolkit.shortcuts import radiolist_dialog

mylist = ["one", "two", "three"]
mylist = list(map(lambda x: (x, x), mylist))
print(mylist)
result = radiolist_dialog(
    title="User selection",
    text="Which user would you like to choose",
    values=mylist,
).run()

# print(result)
