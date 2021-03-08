from utils import iou_mat, CalculateAveragePrecision
import os
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from PyQt5.QtCore import Qt, center
from PyQt5.QtGui import QPaintEvent, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns
sns.set(font_scale=.6)


DEBUG = True
COLORS = np.random.random(size=(14, 3))


def check_im_file(x):
    return os.path.splitext(x)[-1] in ['.jpg', '.png']


def process_bbox(bb, nw, nh):
    x_min = int(bb.x_min * nw)
    x_max = int(bb.x_max * nw)
    y_min = int(bb.y_min * nh)
    y_max = int(bb.y_max * nh)
    return [bb.class_id, x_min, x_max, y_min, y_max]


COORD_LABELS = ['x_min', 'y_min', 'x_max', 'y_max']


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.axes = plt.subplots(2, 1)
        plt.subplots_adjust(hspace=0.5)
        super(MplCanvas, self).__init__(fig)


class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.ann_bboxes = []
        self.ref_bboxes = []

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        self.painter = QPainter()
        self.painter.begin(self)
        self.drawBboxes()
        self.painter.end()

    def drawSingleBbox(self, class_id, x_min, x_max, y_min, y_max, pen):
        self.painter.setPen(pen)
        self.painter.drawRect(x_min, y_min, x_max - x_min, y_max - y_min)
        self.painter.drawText(x_min, y_min+10, str(class_id))

    def drawBboxes(self):
        for class_id, x_min, x_max, y_min, y_max in self.ann_bboxes:
            self.drawSingleBbox(class_id,
                                x_min, x_max,
                                y_min, y_max,
                                QPen(Qt.yellow))

        for class_id, x_min, x_max, y_min, y_max in self.ref_bboxes:
            self.drawSingleBbox(class_id,
                                x_min, x_max,
                                y_min, y_max,
                                QPen(Qt.green))


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1280, 720)

        self.imlist = None
        self.im_id = 0
        self.imdir = ''

        self.meta_df = None

        self.ann_df = None
        self.useann = False

        self.ref_df = None
        self.useref = False

        self.conf = 0.5
        self.clsfilter = list(range(14))

        # ==>

        w_imview = QWidget()
        l_imview = QVBoxLayout()
        w_imview.setLayout(l_imview)

        # ====>

        w_imdir = QWidget()
        l_imdir = QHBoxLayout()
        w_imdir.setLayout(l_imdir)

        w_imdir_btn = QPushButton('Browse')
        w_imdir_btn.setFixedWidth(100)
        w_imdir_btn.clicked.connect(self.btnClicked_imdir)
        l_imdir.addWidget(w_imdir_btn)

        self.w_imdir_txt = QLabel('Choose the directory containing the images')
        l_imdir.addWidget(self.w_imdir_txt)

        l_imview.addWidget(w_imdir)

        # ====>

        w_show = QWidget()
        l_show = QHBoxLayout(w_show)

        w_imshow = QScrollArea(self)
        w_imshow.setFixedWidth(600)
        self.img = ImageView()
        w_imshow.setWidget(self.img)
        l_show.addWidget(w_imshow)

        self.sc = MplCanvas()
        self.format_plot()
        l_show.addWidget(self.sc)

        l_imview.addWidget(w_show)

        # ====>

        self.w_impick = QWidget()
        self.w_impick.setDisabled(True)
        l_impick = QHBoxLayout()
        self.w_impick.setLayout(l_impick)

        w_impick_leftbtn = QPushButton('<')
        w_impick_leftbtn.setFixedWidth(30)
        w_impick_leftbtn.clicked.connect(self.btnClicked_imdir_leftbtn)
        l_impick.addWidget(w_impick_leftbtn)

        self.w_impick_idtxt = QLineEdit()
        self.w_impick_idtxt.setFixedWidth(100)
        self.w_impick_idtxt.returnPressed.connect(self.txtEdited_impick_id_txt)
        l_impick.addWidget(self.w_impick_idtxt)

        w_impick_rightbtn = QPushButton('>')
        w_impick_rightbtn.setFixedWidth(30)
        w_impick_rightbtn.clicked.connect(self.btnClicked_imdir_rightbtn)
        l_impick.addWidget(w_impick_rightbtn)

        self.w_impick_fntxt = QLabel()
        l_impick.addWidget(self.w_impick_fntxt)

        l_imview.addWidget(self.w_impick)

        # ======= Configuration =======

        w_cfg = QWidget()
        w_cfg.setFixedWidth(250)

        l_cfg = QVBoxLayout()
        w_cfg.setLayout(l_cfg)

        # .......

        w_loadcsv = QWidget()
        w_loadcsv.setFixedHeight(150)
        l_loadcsv = QVBoxLayout()
        w_loadcsv.setLayout(l_loadcsv)

        w_metafn = QWidget()
        l_metafn = QHBoxLayout()
        w_metafn.setLayout(l_metafn)

        l_metafn.addWidget(QLabel('Metadata'))

        w_metafn_btn = QPushButton('...')
        w_metafn_btn.setFixedWidth(20)
        w_metafn_btn.clicked.connect(self.btnClicked_metafn)
        l_metafn.addWidget(w_metafn_btn)

        self.w_metafn_txt = QLabel()
        l_metafn.addWidget(self.w_metafn_txt)

        l_loadcsv.addWidget(w_metafn)

        # .......

        w_annfn = QWidget()
        l_annfn = QHBoxLayout(w_annfn)

        self.cb_useann = QCheckBox()
        self.cb_useann.setFixedWidth(20)
        self.cb_useann.stateChanged.connect(self.cbChecked_useann)
        l_annfn.addWidget(self.cb_useann)

        l_annfn.addWidget(QLabel('Prediction'))
        w_annfn_btn = QPushButton('...')
        w_annfn_btn.setFixedWidth(20)
        w_annfn_btn.clicked.connect(self.btnClicked_annfn)
        l_annfn.addWidget(w_annfn_btn)

        self.w_annfn_txt = QLabel()
        l_annfn.addWidget(self.w_annfn_txt)

        l_loadcsv.addWidget(w_annfn)

        # .......

        w_reffn = QWidget()
        l_reffn = QHBoxLayout(w_reffn)

        self.cb_useref = QCheckBox()
        self.cb_useref.setFixedWidth(20)
        self.cb_useref.stateChanged.connect(self.cbChecked_useref)
        l_reffn.addWidget(self.cb_useref)

        l_reffn.addWidget(QLabel('Ground truth'))
        w_reffn_btn = QPushButton('...')
        w_reffn_btn.setFixedWidth(20)
        w_reffn_btn.clicked.connect(self.btnClicked_reffn)
        l_reffn.addWidget(w_reffn_btn)

        self.w_reffn_txt = QLabel()
        l_reffn.addWidget(self.w_reffn_txt)

        l_loadcsv.addWidget(w_reffn)

        l_cfg.addWidget(w_loadcsv)

        # ..........

        w_conf = QWidget()
        w_conf.setFixedHeight(100)

        l_conf = QVBoxLayout()
        w_conf.setLayout(l_conf)

        w_conf_lbltxt = QWidget()
        l_conf_lbltxt = QHBoxLayout(w_conf_lbltxt)

        txt_conf = QLabel('Confidence threshold:')
        l_conf_lbltxt.addWidget(txt_conf)

        self.w_conf_txt = QLineEdit('0.5')
        self.w_conf_txt.setFixedWidth(50)
        self.w_conf_txt.returnPressed.connect(self.txtEdited_conf_txt)
        l_conf_lbltxt.addWidget(self.w_conf_txt)

        l_conf.addWidget(w_conf_lbltxt)

        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setMinimum(0)
        self.sl_conf.setMaximum(10000)
        self.sl_conf.setValue(5000)
        self.sl_conf.setTickInterval(1)
        self.sl_conf.valueChanged.connect(self.slEdited_conf)
        l_conf.addWidget(self.sl_conf)

        l_cfg.addWidget(w_conf)

        # .....

        w_scroll_clsfilter = QScrollArea()
        w_scroll_clsfilter.setFixedHeight(250)
        w_scroll_clsfilter.setWidgetResizable(True)

        w_clsfilter = QWidget()
        w_scroll_clsfilter.setWidget(w_clsfilter)
        l_clsfilter = QVBoxLayout(w_clsfilter)

        self.w_gb_clsfilter = QGroupBox('Class filter')
        l_gb_clsfilter = QVBoxLayout(self.w_gb_clsfilter)
        for i in range(14):
            item = QCheckBox(str(i), self.w_gb_clsfilter)
            item.setChecked(True)
            item.stateChanged.connect(self.cbToggled_clsfilter)
            l_gb_clsfilter.addWidget(item)
        l_clsfilter.addWidget(self.w_gb_clsfilter)

        l_cfg.addWidget(w_scroll_clsfilter)

        # .....

        self.l_root = QHBoxLayout(self)
        self.l_root.addWidget(w_imview)
        self.l_root.addWidget(w_cfg)

    def format_plot(self):
        self.sc.axes[0].cla()
        self.sc.axes[1].cla()

        self.sc.axes[0].set_title('P-R Curve')
        self.sc.axes[0].set_xticks([.0, .2, .4, .6, .8, 1.])
        self.sc.axes[0].set_yticks([.0, .2, .4, .6, .8, 1.])
        self.sc.axes[0].set_xlim(0, 1.1)
        self.sc.axes[0].set_ylim(0, 1.1)

        self.sc.axes[1].set_title('Class AP')
        self.sc.axes[1].set_xticks([.0, .2, .4, .6, .8, 1.])
        self.sc.axes[1].set_yticks(list(range(14)))
        self.sc.axes[1].set_xlim(0, 1.1)
        self.sc.axes[1].set_ylim(-1, 14)

    def cbChecked_useann(self):
        self.useann = self.cb_useann.isChecked()
        self.drawBboxes()

    def cbChecked_useref(self):
        self.useref = self.cb_useref.isChecked()
        self.drawBboxes()

    def cbToggled_clsfilter(self):
        self.clsfilter = [
            int(box.text())
            for box in self.w_gb_clsfilter.findChildren(QCheckBox)
            if box.isChecked()
        ]
        self.notify_clsfilter_changed()

    def slEdited_conf(self):
        self.conf = self.sl_conf.value() * 1.0 / self.sl_conf.maximum()
        self.notify_conf_changed()

    def btnClicked_imdir(self):
        picked = QFileDialog.getExistingDirectory(self, 'Select Folder')

        if DEBUG:
            picked = 'data/train'
        if picked != '':
            self.imdir = picked
            self.w_imdir_txt.setText(self.imdir)
            self.init_imlist = pd.Series(
                filter(check_im_file, os.listdir(self.imdir))
            )
            self.imlist = pd.Series(
                filter(check_im_file, os.listdir(self.imdir))
            )

            self.im_id = 0
            self.notify_imid_changed()

            self.w_impick.setDisabled(False)

    def btnClicked_metafn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)

        if DEBUG:
            picked = 'data/meta/train_info.csv'
        if picked != '':
            self.metafn = picked
            self.w_metafn_txt.setText(os.path.basename(self.metafn))

            self.meta_df = pd.read_csv(self.metafn)
            self.meta_df = self.meta_df.set_index('image_id')

    def btnClicked_annfn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)

        if DEBUG:
            picked = '0.csv'
        if picked != '':
            self.annfn = picked
            self.w_annfn_txt.setText(os.path.basename(self.annfn))

            self.ann_df = pd.read_csv(self.annfn)
            self.ann_df['class_id'] = self.ann_df['class_id'].astype(int)

    def btnClicked_reffn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)

        if DEBUG:
            picked = 'data/raw/folds/0/0_val.csv'
        if picked != '':
            self.reffn = picked
            self.w_reffn_txt.setText(os.path.basename(self.reffn))

            self.ref_df = pd.read_csv(self.reffn)
            self.ref_df['class_id'] = self.ref_df['class_id'].astype(int)

            self.ref_df = self.ref_df.merge(
                self.meta_df,
                left_on='image_id', right_on='image_id',
                suffixes=('_', None)
            )
            self.ref_df.x_min /= self.ref_df.width
            self.ref_df.x_max /= self.ref_df.width
            self.ref_df.y_min /= self.ref_df.height
            self.ref_df.y_max /= self.ref_df.height

            t = self.init_imlist.apply(lambda x: x.split('.')[0])
            self.imlist = self.init_imlist[
                t.isin(self.ref_df['image_id'].values)
            ]
            self.im_id = 0
            self.notify_imid_changed()

    def notify_imid_changed(self):
        self.w_impick_idtxt.setText(str(self.im_id))
        self.w_impick_fntxt.setText(self.imlist.values[self.im_id])
        self.setImage(self.imdir + '/' + self.imlist.values[self.im_id])

        self.drawBboxes()

    def notify_conf_changed(self):
        self.w_conf_txt.setText(f'{self.conf:.4f}')
        self.sl_conf.setValue(int(self.conf * self.sl_conf.maximum()))
        self.drawBboxes()

    def notify_clsfilter_changed(self):
        self.drawBboxes()

    def btnClicked_imdir_leftbtn(self):
        if self.im_id > 0:
            self.im_id -= 1
            self.notify_imid_changed()

    def btnClicked_imdir_rightbtn(self):
        if self.im_id < len(self.imlist) - 1:
            self.im_id += 1
            self.notify_imid_changed()

    def txtEdited_conf_txt(self):
        try:
            new_conf = float(self.w_conf_txt.text())
            if 0 <= new_conf <= 1:
                self.conf = new_conf
        except:
            print('Not a float!')
        self.notify_conf_changed()

    def txtEdited_impick_id_txt(self):
        if self.w_impick_idtxt.text().isdigit():
            new_im_id = int(self.w_impick_idtxt.text())
            if 0 <= new_im_id < len(self.imlist):
                self.im_id = new_im_id
        self.notify_imid_changed()

    def setImage(self, image_fn):
        self.img.setPixmap(QPixmap(image_fn))
        self.img.adjustSize()

    def drawBboxes(self):
        if self.imlist is None or self.meta_df is None:
            return

        pid = os.path.splitext(self.imlist.values[self.im_id])[0]
        nw, nh = self.img.width(), self.img.height()

        if self.useann and self.ann_df is not None:
            pred = self.ann_df[self.ann_df['image_id'] == pid]
            pred = pred[pred.score >= self.conf]
            pred = pred[pred.class_id.isin(self.clsfilter)]

            self.img.ann_bboxes = [
                process_bbox(bb, nw, nh)
                for _, bb in pred.iterrows()
            ]
        else:
            self.img.ann_bboxes = []

        if self.useref and self.ref_df is not None:
            gt = self.ref_df[self.ref_df['image_id'] == pid]
            gt = gt[gt['class_id'].isin(self.clsfilter)]

            self.img.ref_bboxes = [
                process_bbox(bb, nw, nh)
                for _, bb in gt.iterrows()
            ]

            if self.useann and self.ann_df is not None:
                self.format_plot()
                aps = np.zeros(len(self.clsfilter))
                for cid, c in enumerate(self.clsfilter):
                    pic = pred[pred['class_id'] == c]
                    gic = gt[gt['class_id'] == c]
                    ious = iou_mat(pic[COORD_LABELS].values,
                                   gic[COORD_LABELS].values)

                    score_sort_indices = np.argsort(-pic['score'].values)
                    sorted_ious = ious[score_sort_indices]

                    n_p, n_g = ious.shape

                    TP, FP = np.zeros(n_p), np.zeros(n_p)
                    lst_TP_g = []
                    lst_TP_p = []

                    for i in range(n_p):
                        if n_g == 0:
                            FP[i] = 1
                        else:
                            best_iou = sorted_ious[i].max()
                            best_iou_gid = sorted_ious[i].argmax()
                            if best_iou > 0.4:
                                if best_iou_gid in lst_TP_g:
                                    FP[i] = 1
                                else:
                                    TP[i] += 1
                                    lst_TP_g.append(best_iou_gid)
                                    lst_TP_p.append(score_sort_indices[i])
                            else:
                                FP[i] = 1

                    TP = TP.cumsum()
                    FP = FP.cumsum()
                    FN = n_g - TP
                    pre = np.nan_to_num(TP / (TP + FP + 1e-8))
                    rec = np.nan_to_num(TP / (TP + FN + 1e-8))

                    aps[cid] = CalculateAveragePrecision(rec, pre)[0]

                    plt_lbl = f'{cid}' if aps[cid] > 0 else None
                    self.sc.axes[0].plot(rec, pre,
                                         label=plt_lbl,
                                         color=COLORS[c])

                self.sc.axes[0].legend()
                self.sc.axes[1].barh(self.clsfilter, aps, align='center')
                for i, (j, v) in enumerate(zip(self.clsfilter, aps)):
                    if v > 0:
                        self.sc.axes[1].text(v + .01, j - .33, f'{v:.2f}')
                self.sc.draw()
        else:
            self.img.ref_bboxes = []

        self.img.update()


app = QApplication([])
window = MyApp()
window.show()
app.exec()
