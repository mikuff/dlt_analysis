from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Input, Button, DataTable, Label,ListView,ListItem
from textual.containers import Horizontal, Vertical, Container

from utils.cutil import store_dir_access,store_dir_list,check_network,store_file_to_dict,search,search_limit,response_html_to_dict,save_to_store_file
import logging


class DataScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()

        yield Container(
            Vertical(
                # 上方搜索栏
                Horizontal(
                    Label("期数范围:",id="label_pre"),
                    Input(placeholder="开始周期", id="start_input"),
                    Label("~",id="label_mid"),
                    Input(placeholder="结束周期", id="end_input"),
                    Button("检索", id="search_btn"),
                    Button("最近30期", id="search_30_btn"),
                    Button("最近50期", id="search_50_btn"),
                    Button("最近100期", id="search_100_btn"),
                    Button("存储", id="save_btn"),
                    Button("检测网络", id="network_btn"),
                    id="data_top_bar"
                ),

                # 下方主体内容
                Horizontal(

                    # 左侧存储列表
                    Vertical(
                        Label("存储列表", id="left_list_title"),
                        ListView(id="left_list"),
                        Label("Enter(预览)", id="left_list_tip"),
                        id="data_content_left",
                    ),

                    # 右侧
                    Vertical(
                        Label("开奖数据", id="right_table_title"),
                        DataTable(id="right_table"),
                        id="data_content_right",
                    ),
                    id="data_content",
                ),
                id="data_main_vertical"
            ),
            id="data_container"
        )

        yield Footer()

    # 挂载
    def on_mount(self) -> None:
         # 添加右侧表头
        self.__add_right_table_header();

        # 渲染左侧列表数据
        self.__refresh_left_list()

        # 等组件渲染完成后，执行聚焦加载表格操作
        self.call_after_refresh(self.__init_data_pannel)

    # 按钮事件分发
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search_btn":

            start_val = self.query_one("#start_input", Input).value 
            end_val = self.query_one("#end_input", Input).value 

            if not (start_val.isdigit() and end_val.isdigit()):
                self.notify(f'开始周期和结束周期必须是正整数', title="错误", severity="error")
                return
            start, end = int(start_val), int(end_val)
            if start <= 0 or end <= 0 or start > end:
                self.notify(f'开始周期和接受周期必须大于0,且开始周期不能大于结束周期', title="错误", severity="error")
                return
            search_ret = search(start,end)
            self.__init_data_pannel_by_search(search_ret)
        if event.button.id == "search_30_btn":
            search_ret_30 = search_limit(30)
            self.__init_data_pannel_by_search(search_ret_30)
        if event.button.id == "search_50_btn":
            search_ret_50 = search_limit(50)
            self.__init_data_pannel_by_search(search_ret_50)
        if event.button.id == "search_100_btn":
            search_ret_100 = search_limit(100)
            self.__init_data_pannel_by_search(search_ret_100)
        elif event.button.id == "save_btn":
            table = self.query_one("#right_table", DataTable)
            if table.source_type != "source_search" or len(table.source_data) == 0:
                self.notify(f'请先检索数据后再点击存储', title="错误", severity="error")
                return
            if not store_dir_access():
                self.notify(f'请检查程序运行环境,无权限操作用户目录', title="失败", severity="error")
                return
            file_name = save_to_store_file(table.source_data)
            if len(file_name) == 0:
                self.notify(f'存储数据失败', title="失败", severity="error")
                return 
            self.notify(f'存储数据成功 {file_name}', title="成功", severity="information")
            self.__refresh_left_list();
            self.call_after_refresh(self.__init_data_pannel)
                        
        elif event.button.id == "network_btn":
            if check_network():
                self.notify(f'datachart.500.com 检测成功', title="成功", severity="information")
            else:
                self.notify(f'datachart.500.com 检测失败', title="失败", severity="error")
    
    # list 切换事件分发
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.control.id == "left_list":
            item = event.item
            label = item.query_one(Label)
            item_text = label.renderable if label.renderable else ""
            logging.info(f"左侧列表enter选择项{item_text}")
            if len(item_text) > 0 :
                dict_data = store_file_to_dict(item_text)
                if(len(dict_data) == 0):
                    return
                self.__refresh_right_table("source_file",dict_data);

    def __init_data_pannel_by_search(self,search_ret):
        if len(search_ret) == 0 : 
            self.notify(f'检索数据失败', title="失败", severity="error")
            return
        dict_data = response_html_to_dict(search_ret)
        if len(dict_data) == 0 : 
            self.notify(f'解析检索数据失败', title="失败", severity="error")
            return 
        first_issue = dict_data[-1].get("issue","")
        if len(first_issue) > 0:
            self.query_one("#start_input", Input).value = first_issue
        
        last_issue = dict_data[0].get("issue","")
        if len(last_issue) > 0:
            self.query_one("#end_input", Input).value = last_issue

        self.__refresh_right_table("source_search",dict_data);
        self.notify(f'检索成功,{len(dict_data)}数据', title="成功", severity="information")


    # 聚焦左侧列表并渲染数据
    def __init_data_pannel(self):
        # 聚焦
        self.__left_list_activate()
        
        # 获取焦点
        active_file_name = self.__get_left_list_active()
        if(len(active_file_name) == 0):
            return 
        logging.info(f"左侧列表激活选项是:{active_file_name}")
        
        # 加载数据
        dict_data = store_file_to_dict(active_file_name)
        if(len(dict_data) == 0):
            return
        
        # 渲染右侧表格
        self.__refresh_right_table("source_file",dict_data);


    # 获取文件本地文件列表
    def __local_file_list(self) -> list[str]:
        if not store_dir_access():
            self.notify(f'请检查程序运行环境,无权限操作用户目录', title="失败", severity="error")
            return [];
        return store_dir_list()
    
    # 添加右侧表格表头
    def __add_right_table_header(self) -> None:
        table: DataTable = self.query_one("#right_table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True

        table.add_column("期号", width=30, key="issue")
        for i in range(1, 6):
            table.add_column(f"前区_{i}", width=15, key=f"rb_{i}")
        for i in range(1, 3):
            table.add_column(f"后区_{i}", width=15, key=f"bb_{i}")
        table.add_column("开奖日期", width=30, key="date")
    

    # 刷新左侧存储列表
    def __refresh_left_list(self) -> None:
        data = self.__local_file_list()
        if len(data) <= 0:
            return
        left_list = self.query_one("#left_list", ListView)
        left_list.clear()
        for item_text in data:
            left_list.append(ListItem(Label(item_text)))
    
    def __left_list_activate(self):
        left_list = self.query_one("#left_list", ListView)
        if left_list.children:
            left_list.index = 0
            left_list.focus()

    # 获取左侧焦点所在的文件名
    def __get_left_list_active(self) -> str:
        left_list = self.query_one("#left_list", ListView)
        index = left_list.index

        if index is None or index < 0 or index >= len(left_list.children):
            return ""

        list_item = left_list.children[index]
        label = list_item.query_one(Label)

        return label.renderable if isinstance(label.renderable, str) else str(label.renderable)

    # 刷新右侧表格
    def __refresh_right_table(self, source_type: str, data: list[dict]) -> None:
        
        label = self.query_one("#right_table_title", Label)
        if source_type == "source_search":
            label.update("开奖数据(检索结果)")
        elif source_type == "source_file":
            label.update("开奖数据(存储内容)")
        else:
            label.update("开奖数据")

        table = self.query_one("#right_table", DataTable)
        table.clear()

        for row in data:
            row_data = [
                row.get("issue", ""),
                row.get("rb_1", ""),
                row.get("rb_2", ""),
                row.get("rb_3", ""),
                row.get("rb_4", ""),
                row.get("rb_5", ""),
                row.get("bb_1", ""),
                row.get("bb_2", ""),
                row.get("date", ""),
            ]
            table.add_row(*row_data)

        table.cursor_type = "row"
        table.zebra_stripes = True
        table.source_type =source_type
        table.source_data = data


