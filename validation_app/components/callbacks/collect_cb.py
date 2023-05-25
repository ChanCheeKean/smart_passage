import os
import cv2
import time
from datetime import datetime
from glob import glob
import base64
from dash import html, Input, Output, State, no_update, callback_context
from app import app, video_loader

# initialize oringal path
data_path = os.path.join(".", "data", "landing")

### libraries dropdown
@app.callback(
    [
        Output("collect_library_dropdown", "options"),
        Output("collect_library_dropdown", "value"),
        ],
    Input("url", "pathname"),
)
def return_dropdown_options(pth):
    dir_list = glob(os.path.join(data_path, "*"), recursive=True)
    # dir_list = [x.split("/")[-2] for x in dir_list]
    dir_list = [os.path.split(x)[-1] for x in dir_list]
    options = [{'label': x,'value': x} for x in dir_list]
    return options, dir_list[0]


### save image if click button
@app.callback(
    Output("dummy_div", "children"),
    [
         Input("capture_but", "n_clicks"), 
         Input("delete_but", "n_clicks")
    ],
    State("collect_library_dropdown", "value")
)
def save_image(n1, n2, lib):

    # dont run during initialization
    if (n1 == 0) and (n2 == 0):
        return no_update
    
    # make directory if not in there
    folder_path = os.path.join(data_path, lib)
    os.makedirs(folder_path, exist_ok=True)

    # save image
    if callback_context.triggered_id == 'capture_but':
        loader = video_loader
        _, image = loader.read()
        image = cv2.flip(image, 1)
        img_name = f'{int(time.time())}.jpg'
        print(f"Saving Image: {img_name}")
        cv2.imwrite(os.path.join(folder_path, img_name), image)

    # delete last image
    elif callback_context.triggered_id == 'delete_but':
         files = [f for f in os.listdir(folder_path) if ".jpg" in f]
         if files:
            sort_files = sorted([f.split('.')[0] for f in files], reverse=True)
            os.remove(os.path.join(folder_path, f"{sort_files[0]}.jpg"))
            print(f"Deleted Image: {sort_files[0]}")

    return None


### show image if click button
@app.callback(
    Output("img_archive", "children"),
    Input("dummy_div", "children"),
    Input("collect_library_dropdown", "value")
)
def show_image(n1, lib):
    folder_path = os.path.join(data_path, lib)
    files = [f for f in os.listdir(folder_path) if ".jpg" in f]
    if files:
        sort_files = sorted([f.split('.')[0] for f in files], reverse=True)
        child_list = []
        for f in sort_files[:5]:
            img_pth = os.path.join(folder_path, f'{f}.jpg')
            with open(img_pth, "rb") as img_data: 
                img_data = base64.b64encode(img_data.read()).decode()                
            child_list.append(
                html.Img(src=f"data:image/jpg;base64,{img_data}", alt='image', width=400, height=240, className='my-1')
            )
        return child_list
    else:
        return None