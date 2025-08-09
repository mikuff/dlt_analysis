from textual.screen import Screen
from textual.widgets import Header,Footer,Log,Select,Label,Input,Button
from textual.widgets import Footer
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import TextArea
from utils.cutil import store_dir_access,store_dir_list,check_input_in_range,is_valid_json,check_file_exist,get_abspath
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import logging
import json
import threading

from utils.model_util import LotteryPredictor

class AnalysisScreen(Screen):

    DEFAULT_MODEL_CONFIG = {
        'lightgbm':{
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "num_leaves": 16,
            "min_child_samples": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1
        },
        'random_forest': {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": 0
        }
    }

    def on_mount(self) -> None:
        # 检查权限
        if not store_dir_access():
            self.notify(f'请检查程序运行环境,无权限操作用户目录', title="失败", severity="error")
            return;
         # 初始化select选择框
        file_list = store_dir_list()
        options = [(d, d) for d in file_list] 
        self.query_one("#select_file").set_options(options);
        self.log_widget  = self.query_one("#left_preview");


    def compose(self):
        yield Header()
        yield Container(
            Horizontal(
                Vertical(
                    Label("模型配置", id="right_form_title"),
                    Select(prompt="数据文件",options=[],id="select_file"),
                    Horizontal(
                        Select(prompt="模型类型",options=[("random_forest", "random_forest"),("lightgbm", "lightgbm")],id="model_type"),
                        Select(prompt="启用特征工程",options=[("是", "是"),("否", "否")],id="feature_engineering"),
                        id="model"
                    ),
                    Horizontal(
                        Input(placeholder="回溯期数", id="lookback_periods"),
                        Input(placeholder="测试集比例", id="test_size"),
                        id="model_config"
                    ),
                    TextArea.code_editor("", language="json",id="model_param"),
                    Horizontal(
                        Button("默认/重置", id="default_btn"),
                        Button("执行", id="exec_btn"),
                        id="btn_group"
                    ),
                    id="right_form"),
                Log(id="left_preview",auto_scroll=True),
            ),
            id="analysis_container"
        )
        yield Footer()
    
    # btn执行分发
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'default_btn':
            self.call_after_refresh(self.__reset_form)
        if event.button.id == 'exec_btn':
            (bl,msg,json) = self.__check_config();
            if not bl:
                self.notify(f'模型配置错误,{msg}', title="失败", severity="error")
                return
            else:
                self.exec_model(json)

    # select执行分发
    def on_select_changed(self, message: Select.Changed) -> None:
        if message.select.id == 'model_type':
            self.__update_model_param()

    # 重置表单
    def __reset_form(self):
        dirs = store_dir_list()
        if dirs:
            self.query_one("#select_file").value = dirs[0]
        self.query_one("#model_type").value = 'random_forest'
        self.query_one('#feature_engineering').value = '是'
        self.query_one('#lookback_periods').value = '15'
        self.query_one('#test_size').value = '0.3'
        self.__update_model_param();
    
    # 根据模型更新模型参数
    def __update_model_param(self):
        try:
            model_type = self.query_one("#model_type").value
            mode_param = self.DEFAULT_MODEL_CONFIG[model_type]
            self.query_one('#model_param').text = json.dumps(mode_param,indent=4,ensure_ascii=False);
        except:
            self.query_one('#model_param').text = json.dumps({},indent=4,ensure_ascii=False);

    # 检查模型参数配置
    def __check_config(self):
        select_file = self.query_one("#select_file").value
        if type(select_file).__name__ == "NoSelection":
            return (False,"请选择数据文件",{});
        model_type = self.query_one("#model_type").value
        if type(model_type).__name__ == "NoSelection":
            return (False,"请选择模型类型",{});
        feature_engineering = self.query_one("#feature_engineering").value
        if type(feature_engineering).__name__ == "NoSelection":
            return (False,"请选择是否启用特征工程",{});
        lookback_periods = self.query_one("#lookback_periods").value.strip()
        bl,lookback_periods_num = check_input_in_range(lookback_periods, 1, 100, integer=True)
        if not bl:
            return (False,"请设置合理的回溯期数,合理范围(1-100)",{});
        lookback_periods = lookback_periods_num;
        test_size = self.query_one("#test_size").value.strip()
        (bl,test_size_num) = check_input_in_range(test_size, 0.1, 0.9, integer=False)
        if not bl:
            return (False,"请设置合理的回溯期数,合理范围(0.1-0.9)",{});
        test_size = test_size_num;
        model_param = self.query_one('#model_param').text
        if not is_valid_json(model_param):
            return (False,"请检查模型参数",{});
        return (True,"",{
            "select_file":select_file,
            "model_type":model_type,
            "feature_engineering":feature_engineering,
            "lookback_periods":lookback_periods,
            "test_size":test_size,
            "model_param": json.loads(model_param)
        })
    
    def safe_render(self,message: str):
        self.call_after_refresh(self.log_widget.write_line, message)

    # 执行模型
    def exec_model(self,config):

        select_file = config.get('select_file')
        model_type = config.get('model_type')
        feature_engineering = config.get('feature_engineering')
        lookback_periods = config.get('lookback_periods')
        test_size = config.get('test_size')
        model_param = config.get('model_param')

        if not check_file_exist(select_file):
            self.notify(f'请检查文件是否存在或是否有权限读取 {select_file}', title="失败", severity="error")
            return 
        model = self.__model_builder(model_type,model_param)
        if model is None:
            self.notify(f'模型参数异常', title="失败", severity="error")
            return 
        feature_engineering_bl = feature_engineering == "是"

        try:
            self.query_one('#exec_btn').disabled = True

            predictor = LotteryPredictor(lookback_periods=lookback_periods,
                model_type=model_type,
                verbose=True,
                feature_engineering=feature_engineering_bl,
                test_size=test_size,
                model=model,
                render = self.safe_render,
                file_path= get_abspath(select_file)
            )
            
            threading.Thread(target=predictor.start, daemon=True).start()

            self.query_one('#exec_btn').disabled = False
        except Exception as e:
            self.query_one('#exec_btn').disabled = False
            logging.info(f"运行模型失败 {e}")            
    
    # 模型构建
    def __model_builder(self,model_type: str, model_param: dict):
        try:
            if model_type == "random_forest":
                return RandomForestClassifier(
                    n_estimators=model_param["n_estimators"],
                    max_depth=model_param["max_depth"],
                    min_samples_split=model_param["min_samples_split"],
                    min_samples_leaf=model_param["min_samples_leaf"],
                    random_state=model_param["random_state"],
                    n_jobs=model_param["n_jobs"],
                    verbose=model_param["verbose"],
                )
            elif model_type == "lightgbm":
                return LGBMClassifier(
                    n_estimators=model_param["n_estimators"],
                    max_depth=model_param["max_depth"],
                    learning_rate=model_param["learning_rate"],
                    num_leaves=model_param["num_leaves"],
                    min_child_samples=model_param["min_child_samples"],
                    subsample=model_param["subsample"],
                    colsample_bytree=model_param["colsample_bytree"],
                    reg_alpha=model_param["reg_alpha"],
                    reg_lambda=model_param["reg_lambda"],
                    random_state=model_param["random_state"],
                    n_jobs=model_param["n_jobs"],
                    verbose=model_param["verbose"],
                )
            else:  
                return None
        except Exception as e:
            logging.error(f"模型构建失败，异常: {e}")
            return None 


        


    