import os, base64
from datetime import datetime
from glob import glob
import shutil
import json
from dash import Input, Output, State, no_update, callback_context
from app import app

# initialize data path
data_path = os.path.join(".", "data", "landing")
export_data_path = os.path.join(".", "data", "processed")
if not os.path.exists(export_data_path):
    os.makedirs(export_data_path)

### libraries dropdown ###
@app.callback(
    [
        Output("library_dropdown", "options"),
        Output("library_dropdown", "value"),
        ],
    Input("url", "pathname"),
)
def return_dropdown_options(_):
    # get the list of subfolders
    dir_list = glob(os.path.join(data_path, "*"), recursive=True)
    dir_list = [os.path.split(x)[-1] for x in dir_list]
    options = [{'label': x,'value' : x} for x in dir_list]
    return options, dir_list[0]

### show image carousel ###
@app.callback(
    [
        Output("img_carousel", "items"),
        Output("img_carousel", "active_index"),
        ],
    Input("library_dropdown", "value"),
    Input("label_switch", "value"),
)
def show_img(lib, switch_bool):

    if lib is None:
        return no_update

    # get all the files in the subfolder
    folder_path = os.path.join(data_path, lib)
    files = [f for f in os.listdir(folder_path) if (".jpg" in f)]

    # if switch is on, filter labeled data
    if switch_bool:
        export_img_path = os.path.join(export_data_path, lib, 'images')
        export_files = os.listdir(export_img_path)
        if export_files:
            files = [f for f in files if f not in export_files]

    # display all the images in sorted manner
    items = []
    if files:
        sort_files = sorted([f.split('.')[0] for f in files], reverse=True)
        for i, f in enumerate(sort_files):
            img_pth = os.path.join(folder_path, f'{f}.jpg')
            with open(img_pth, "rb") as img_data:
                img_data = base64.b64encode(img_data.read()).decode() 
            items.append(
                {
                    "key": i, 
                    "src": f"data:image/jpg;base64,{img_data}", 
                    "img_style": {'height' : '88vh'},
                    "caption": f,
                    })
    return items, 0

### show image plotter
# @app.callback(
#     Output("annotate_img", "figure"),
#     Input("img_carousel", "active_index"),
#     Input("library_dropdown", "value"),
#     prevent_initial_call=True,
# )
# def plot_image(ind, lib):
#     ind = 0 if ind is None else ind
#     folder_path = os.path.join(data_path, lib)
#     files = [f for f in os.listdir(folder_path) if ".jpg" in f]
#     if files:
#         sort_files = sorted([f.split('.')[0] for f in files], reverse=True)
#         img_pth = os.path.join(folder_path, f'{sort_files[ind]}.jpg')
#         img_data = io.imread(img_pth)
#         fig = px.imshow(img_data)
#         fig.update_layout(dragmode="drawrect")
#         return fig
        
#     return no_update

### return the carousel index ###
@app.callback(
    Output("img_annotate_header", "children"),
    Input("img_carousel", "active_index"),
    Input("img_carousel", "items"),
)
def return_index(ind, item):

    # assign ind is 0
    ind = 0 if ind is None else ind
    sort_files = [it['caption'] for it in item]
    
    if len(sort_files) > 0:
        return f"Image Annotatation: {sort_files[ind]}"
    return no_update

### save json
@app.callback(
    Output("textinput_savedjson", "value"),

    Input("save_annotate_but", "n_clicks"), 
    Input("append_annotate_but", "n_clicks"), 
    Input("delete_annotate_but", "n_clicks"), 
    Input("img_carousel", "active_index"),
    Input("img_carousel", "items"),
    State("class_dropdown", "value"),
    State("zone_dropdown", "value"),
    State("textinput_desc", "value"),
    State("library_dropdown", "value"),
)
def return_index(n1, n2, n3, ind, item, class_in, zone_in, text_in, lib):
    if lib is None:
        return no_update
    # defined the export path for both images and labels
    ind = 0 if ind is None else ind
    folder_path = os.path.join(data_path, lib)
    export_img_path = os.path.join(export_data_path, lib, 'images')
    export_text_path = os.path.join(export_data_path, lib, 'labels')
    sort_files = [it['caption'] for it in item]

    # create dir if it's not there
    if not os.path.exists(export_img_path):
        os.makedirs(export_img_path)
        print(f"Deleted {export_img_path}")
    if not os.path.exists(export_text_path):
        os.makedirs(export_text_path)
        print(f"Deleted {export_text_path}")

    # extract the current image name
    img_pth, txt_pth = '', ''
    if len(sort_files) > 0:
        curr_file = sort_files[ind]
        img_pth = os.path.join(export_img_path, f"{curr_file}.jpg")
        txt_pth = os.path.join(export_text_path, f"{curr_file}.json")

    json_data = ''
    new_data = {'class': class_in, 'zone': zone_in, 'tag': text_in}
    if os.path.exists(txt_pth):
        with open(txt_pth, 'r') as f:
            json_data = json.load(f)

    # delete files
    if (n1 == 0) and (n2 == 0) and (n3 == 0) and callback_context.triggered_id != 'img_carousel':
        return json_data

    elif callback_context.triggered_id == 'delete_annotate_but':
        if os.path.exists(img_pth):
            os.remove(img_pth)
        if os.path.exists(txt_pth):
            os.remove(txt_pth)
        return ""
    
    elif callback_context.triggered_id == 'save_annotate_but':
        with open(txt_pth, 'w') as f:
            json.dump([new_data], f)

        shutil.copyfile(
            f'{os.path.join(folder_path, curr_file)}.jpg', 
            img_pth, 
        )
        return str([new_data])
    
    elif callback_context.triggered_id == 'append_annotate_but':
        if json_data != '':
            json_data.append(new_data)
            print(json_data)
        else:
            json_data = [new_data]
        
        with open(txt_pth, 'w') as f:
            json.dump(json_data, f)

        shutil.copyfile(
            f'{os.path.join(folder_path, curr_file)}.jpg', 
            img_pth, 
        )
        return str(json_data)

    return str(json_data)