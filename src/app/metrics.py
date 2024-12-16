import numpy as np
import nibabel as nib
import pydicom as pdcm
import os
from skimage.measure import label,regionprops_table
import pandas as pd
import sys

'''
    Genral function to automatically read our filetypes
    Copide because Python cannot do relative path imports 
    this module will be called from another python file
'''
def read_scan(path:str) -> np.ndarray:

    if '.gz' in os.path.splitext(path)[1]:
        return nib.load(path).get_fdata()
    elif '.dcm' in os.path.splitext(path)[1]:
        return pdcm.dcmread(path).pixel_array
    else:
        return np.load(path,allow_pickle=True)
    

'''
    If image is float mask out close to predicted label
    Just a placeholder for now
'''
def create_mask(scan:np.ndarray) -> np.ndarray:
    return scan

'''
    Make dummy prediction data for testing
'''
def create_dummy_mask(scan:np.ndarray):
    from skimage.draw import ellipse
    rr,cc = ellipse(50,50,r_radius=10,c_radius=15)
    rr1,cc1 = ellipse(100,125,r_radius=15,c_radius=5)
    mask = np.zeros(scan.shape)
    mask[rr,cc] = 1
    mask[rr1,cc1] = 1
    return mask

'''
    Return Region Properties for predicted mask
'''
def get_metrics(prediction:np.ndarray) -> pd.DataFrame:
    scan = prediction
    mask = create_mask(scan)
    segments = label(mask)
    regions = pd.DataFrame(regionprops_table(
        segments,properties=[
            "area",
            "axis_major_length",
            "axis_minor_length",
            "bbox",
            "centroid",
            "label"
        ]
    ))
    #Calculate extra regions
    regions['eccentricity'] = regions['axis_major_length'] / regions["axis_minor_length"]
    
    return regions.rename(
        columns={
            'bbox-0': 'bbox-y0',  # Top-left Y
            'bbox-1': 'bbox-x0',  # Top-left X
            'bbox-2': 'bbox-y1',  # Bottom-right Y
            'bbox-3': 'bbox-x1',  # Bottom-right X
            'centroid-0': 'centroid-y',  # Centroid Y
            'centroid-1': 'centroid-x'  # Centroid X
        }
    )
    
'''
    Conversion code to enable frontend boundary drawing
    changes topleft (x,y) bottomright (x,y) to all four points
    in counter-clockwise order
'''
def metrics_to_bboxes(metrics:pd.DataFrame):
    top_lefts = zip(metrics['bbox-x0'],metrics['bbox-y0'])
    top_rights = zip(metrics['bbox-x1'],metrics['bbox-y0'])
    bottom_lefts = zip(metrics['bbox-x0'],metrics['bbox-y1'])
    bottom_rights = zip(metrics['bbox-x1'],metrics['bbox-y1'])

    return [
        {
            "label":f"Tumor {i}",
            "points":[
                {'x':x1,"y":y1},
                {'x':x2,"y":y2},
                {'x':x3,"y":y3},
                {'x':x4,"y":y4}
                
            ]
        } for i, ((x1,y1),(x2,y2),(x3,y3),(x4,y4)) in enumerate(zip(top_lefts,top_rights,bottom_rights,bottom_lefts))
    ]

def df_to_json(input_data):
    # Parse the table JSON string into a Python dictionary
    table_data = input_data['table']
    
    # Convert the table dictionary into the desired list of dictionaries
    table_transformed = []
    for key, values in table_data.items():
        
        # Check if values has a .values() method (i.e., is dictionary)
        # Restrict floating value to 5 decimals only.
        if hasattr(values, 'values'):
            value_string = ", ".join(
                f"{v:.5f}".rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
                for v in values.values
            )
        # Fallback for non-iterable values
        else:
            value_string = f"{values:.5f}".rstrip('0').rstrip('.') if isinstance(values, float) else str(values)

        table_transformed.append({
            "name": key,
            "value": value_string
        })

    # Transform the shapes with updated points
    shapes_transformed = []
    for shape in input_data['shapes']:
        shapes_transformed.append({
            "label": shape['label'],
            "points": [{"x": point["x"], "y": point["y"]} for point in shape['points']]
        })

    # Assemble the final transformed data
    transformed_data = {
        "original_image": input_data['original_image'],
        "prediction": input_data['prediction'],
        "table": table_transformed,
        "shapes": shapes_transformed
    }

    return transformed_data

if __name__ =="__main__":
    path = r"H:\data_splits\data_splits\test_data\LIDC-IDRI-0002_N000_S171_Mask.npy"
    metrics = get_metrics(path)
    print(metrics)
    scan = read_scan(path)
    mask = np.zeros(scan.shape)
    mask[int(metrics['bbox-x0']):int(metrics['bbox-x1']),int(metrics['bbox-y0']):int(metrics['bbox-y1'])] = 1
    print(metrics_to_bboxes(metrics))
    import matplotlib.pyplot as plt 
    plt.subplot(1,2,1)
    plt.imshow(scan)
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.show()

