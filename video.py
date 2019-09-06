import yaml


from video_facenet.pipelines import process_video
from video_facenet.cluster_analysis import show_clusters


def load_config(file_name):
    with open(file_name, 'r') as stream:
        return yaml.safe_load(stream)

     

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process video using tensorflow facenet')
    API = {
        "show_clusters":show_clusters,
        "process_video":process_video
    }

    parser.add_argument('--task', "-t", type=str, default='current', help='task name in configuration file')        
    args = parser.parse_args()

    tasks = load_config("tasks.yaml")
    task = tasks[args.task]
    func_name = task.get("task", "show_clusters")
    func = API.get(func_name)
    if not func:
        print ("no task <{}> was defined, exiting")
        exit(1)
    if "config" in task:
        config = tasks[task["config"]]
    else:
        config = task

    func(**config)




 
    
