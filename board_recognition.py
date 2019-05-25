"""
Reversi board recognizer
"""

__author__ = "lavox"
__copyright__ = "Copyright (c) 2019 lavox"
__license__ = "MIT License"
__version__ = "1.0"

import numpy as np
import cv2
import functools
import math
from enum import IntEnum

### 入出力用のクラス・Enumの定義 ###
class DiscColor(IntEnum):
    """
    石の色用
    """
    BLACK = 0
    WHITE = 1

class Mode(IntEnum):
    """
    認識モード用
    """
    VIDEO = 0
    PHOTO = 1

class RecognizerType(IntEnum):
    """
    認識クラス用
    """
    REALBOARD = 0
    SCREENSHOT = 1
    PRINTED = 2

class Hint():
    """
    認識するにあたってアプリ側から与えるヒント情報
    """
    def __init__(self):
        # 35mm換算焦点距離[float]
        self.focal = None
        # 画像上の中心点[(int,int)]
        self.center = None
        # 認識モード[Mode]
        self.mode = None

class Disc():
    """
    石の情報格納用
    """
    def __init__(self):
        # 石の色[DiscColor]
        self.color = None
        # 石の座標。盤の左上原点で単位は盤の1辺[(float, float)]Y座標、X座標の順
        self.position = None
        # 対応する盤のマスの位置[(int, int)]Y座標、X座標の順
        self.cell = None

class Result():
    """
    認識結果
    """
    def __init__(self):
        # 検出した石の情報[List[Disc]]
        self.disc = []
        # 不明マス(手などの障害物が写っている等)[boolの2次元配列]
        self.isUnknown = np.array([[False] * 8 for i in range(8)])

        # 写真上での盤の頂点の座標[List[(float, float)]]
        self.vertex = []

        # カメラ位置を原点とした時の盤の4頂点の3次元座標(カメラ座標系。単位=盤の1辺)[List[(float, float, float)]]
        self.vertex3d = []

        # 画像変換後の左上を原点とした時のカメラ位置を盤上に投影した点の2次元座標(単位px)[(float, float)]
        # 実際の盤の写真だと、斜めから撮った際に石の厚みで石がきれいな円にならないので、
        # 天面や底面を判定するために使用する。Y座標、X座標の順
        self.cameraPosition_px = None

        # 盤の中心を原点とした時のカメラ位置(単位=盤の1辺)[(float, float, float)]
        # AR処理で使用することを想定(本モジュール外)
        self.cameraPosition_bd = None

        # どのRecognizerで処理を行ったかを表す
        # AutomaticRecognizerでanalyzeまたはdetectBoardした場合のみ設定される
        self.recognizerType = None

    def clearDiscInfo(self):
        self.disc = []
        self.isUnknown = np.array([[False] * 8 for i in range(8)])

###
### ユーティリティ的な関数の定義
###
def intersection(line0, line1):
    """
    2直線の交点を求める
    引数の直線は、直線上の2点が与えられているものとする
    (参照サイト http://imagingsolution.blog107.fc2.com/blog-entry-137.html)
    """
    S1 = float(np.cross(line0[1] - line0[0], line1[0] - line0[0]))
    S2 = float(np.cross(line0[1] - line0[0], line0[0] - line1[1]))
    return line1[0] + (line1[1] - line1[0]) * S1 / (S1 + S2)

def expand(vtx, ratio):
    """
    4頂点を、対角線の中心を起点にratioの比率だけ広げる
    盤の頂点を求める際に、実盤だと緑色の部分を拾ってしまうので、枠線部分まで範囲を少し
    広げるために使用することを想定
    """
    # 対角線の交点を求める
    center = intersection([vtx[0],vtx[2]], [vtx[1],vtx[3]])
    return list(map(lambda v: (v - center) * ratio + center, vtx))

def argmax(matrix):
    """
    行列の成分で最大の値の座標を求める
    """
    return np.unravel_index(matrix.argmax(), matrix.shape)

def getParallelogramDiagonal(v0, v1, vc):
    """
    3次元座標内に存在し、直線上に並んでいる3点v0,vc,v1(位置ベクトル)に対して、
    v0'=a*v0, v1'=b*v1, vcはv0'とv1'の中点となるようなv0',v1'を求める(a,bはスカラー)
    写真上の盤の対角の頂点v0,v1と、対角線の交点vcがわかっている時に、3次元空間のどこの点から
    投影されたかを求めるという意味。
    """
    # 各ベクトルの長さを算出
    n_v0 = np.linalg.norm(v0)
    n_v1 = np.linalg.norm(v1)
    n_vc = np.linalg.norm(vc)
    
    # v0〜vc間、v1〜vc間の角度のcosを算出
    cos_t0 = np.dot(v0, vc)/(n_v0 * n_vc)
    cos_t1 = np.dot(v1, vc)/(n_v1 * n_vc)
    # sinの値を計算
    sin_t0 = np.sqrt(1.0 - (cos_t0 ** 2))
    sin_t1 = np.sqrt(1.0 - (cos_t1 ** 2))
    # 幾何的な考察により、求めるv0',v1'は、定数kを使って、
    # v0' = k * sin_t1 * v0/n_v0,
    # v1' = k * sin_t0 * v1/n_v1
    # と書ける
    vc0 = sin_t1 * v0 / n_v0 + sin_t0 * v1 / n_v1
    # と置けば、v0'とv1'の中点は、k * vc0 / 2と書ける

    # これがvcに一致することからkの値を求める
    k = np.sqrt((2 * n_vc) ** 2 / (np.linalg.norm(vc0) ** 2))
    return k * sin_t1 * v0 / n_v0, k * sin_t0 * v1 / n_v1

def getParallelogramRatio(vtx, center, img_size, focal, img_center):
    """
    写真の35mm換算焦点距離がfocalだと仮定した時に、写真上のvtxの4点が平行四辺形となるような
    3次元空間上の4点vtx3d(カメラpx座標系)と、短辺と長辺の比rと、その間の角度radを求める
    vtx: 写真上の4点(2次元座標)
    center: vtxの4点の対角線の交点
    img_size: 写真のサイズ(px)
    focal: 35mm換算焦点距離
    img_center: 画像の中心点(カメラの正面の点)
    """
    height, width, _ = img_size

    # 与えられた写真の場合に、焦点距離に対応するカメラからの距離(px)を求める
    z0 = float(focal) * np.sqrt(width ** 2 + height ** 2) / np.sqrt(24 ** 2 + 36 ** 2)
    # 画像の中心点をx0,y0とする。
    if img_center is None:
        x0 = float(width) / 2.0
        y0 = float(height) / 2.0
    else:
        y0, x0 = img_center

    # カメラの3次元上の座標を原点、写真平面がz = z0とした場合、
    # 写真の座標系での点(x,y)は、この3次元空間上は(x-x0, y-y0, z0)となる
    # この座標系は、カメラの向きをz軸の正の方向とし、写真とx,yの方向は合っているものとする

    # 交点の位置ベクトルを求める
    vc = np.array([center[0] - x0, center[1] - y0, z0], dtype=np.float32)

    # 1組目の対角線の各頂点への位置ベクトル
    v0 = np.array([vtx[0][0] - x0, vtx[0][1] - y0, z0], dtype=np.float32)
    v2 = np.array([vtx[2][0] - x0, vtx[2][1] - y0, z0], dtype=np.float32)
    # 対角線の頂点に変換
    v0_d, v2_d = getParallelogramDiagonal(v0, v2, vc)

    # 2組目の対角線の各頂点への位置ベクトルについても同様の計算を行う
    v1 = np.array([vtx[1][0] - x0, vtx[1][1] - y0, z0], dtype=np.float32)
    v3 = np.array([vtx[3][0] - x0, vtx[3][1] - y0, z0], dtype=np.float32)
    v1_d, v3_d = getParallelogramDiagonal(v1, v3, vc)

    # 縦横の長さの比を求める
    ratio = np.linalg.norm(v1_d - v0_d) / np.linalg.norm(v3_d - v0_d)
    # 角度を求める
    rad = math.acos(np.dot(v1_d - v0_d, v3_d - v0_d) / (np.linalg.norm(v1_d - v0_d) * np.linalg.norm(v3_d - v0_d)))
    return ratio, rad, np.array([v0_d, v1_d, v2_d, v3_d])

def getRidgeEdge(distComponent, maxCoord, direction):
    """
    最大値〜最大値-1の範囲で、指定された方向から見て最も遠い点と近い点を見つける。
    緑領域からの距離が最大値近辺で、カメラから見て最も遠い点と近い点を見つけるための関数。
    これにより、石の天面の中心と底面の中心を求める
    """
    # 最大値
    maxValue = distComponent[maxCoord]
    # 最大値-1以上の点の座標群
    ridge = np.array(np.where(distComponent >= maxValue - 1)).T
    # 隣の石を検出しないよう、maxCoordからの距離がmaxValue以内という制約を設ける
    ridge = ridge[np.apply_along_axis(
        lambda pt: np.linalg.norm( np.array(pt) - maxCoord ) <= maxValue , 
        axis=1, 
        arr=ridge)]
    # 内積の値
    dotValue = np.apply_along_axis(
        lambda pt: np.dot(np.array(pt) - maxCoord, direction),
        axis=1,
        arr=ridge
    )
    # 内積が最大になる点の座標と最小になる点の座標を返す
    maxEdgePoint = np.array(ridge[np.argmax(dotValue)])
    minEdgePoint = np.array(ridge[np.argmin(dotValue)])
    return maxEdgePoint, minEdgePoint

#
# 定数値
#
_KERNEL3 = np.ones((3, 3), dtype=np.int)
_KERNEL5 = np.ones((5, 5), dtype=np.int)
_KERNEL9 = np.ones((9, 9), dtype=np.int)

###
### 認識用のクラスの定義
###
### 構成
### BaseRecognizer (基底クラス。直接は使用しない)
###  +- RealBoardRecognizer (実盤用)
###  |   +- ScreenshotRecognizer (スクリーンショット用)
###  +- PrintedBoardRecognizer (書籍等の白黒印刷物用)
###  +- AutoBoardRecognizer (上記のクラスを使って自動検知する)
class BaseRecognizer:
    """
    認識のための基底クラス
    """

    # 認識時に盤の画像を切り出す際のサイズの定義(単位:px)
    # 1マスの辺の長さ(公式盤の1mm=1pxくらい)
    _CELL_SIZE = 42
    # 周囲の余白(余白をつけておいた方が縁付近の特殊な考慮が不要になるので)
    _BOARD_MARGIN = 13
    # 切り出し後の画像のサイズ
    _EXTRACT_IMG_SIZE = _CELL_SIZE * 8 + _BOARD_MARGIN * 2

    # 盤面抽出時の近似の係数
    _EPSILON_COEFF = 0.004

    def analyzeBoard(self, image, hint):
        """
        盤の範囲の認識(detect_board)と石の位置・色の認識(detect_disc)を実行

        image: ndarrayで元となる画像を指定
        hint: 認識時に使用するHint情報
        戻り値: 認識成否(bool)と、成功した場合は結果情報(Result)
        """

        # 盤の検出処理
        ret, result = self.detectBoard(image, hint)
        if ret:
            # 成功した場合は石の検出処理
            return self.detectDisc(image, hint, result)
        else:
            return False, None

    def detectBoard(self, image, hint):
        """
        盤の範囲の認識

        image: ndarrayで元となる画像を指定
        hint: 認識時に使用するHint情報
        戻り値: 認識成否(bool)と、成功した場合は結果情報(Result)
        """

        # 盤の凸包候補を取得する
        ret, hull = self._detectConvexHull(image)
        if ret and hull.shape[0] >= 4:
            # 成功し、かつ4点以上ある場合は結果の設定処理
            return self._resultForDetectBoard(image.shape, hint, hull)
        else:
            return False, None

    def detectDisc(self, image, hint, result):
        """
        石の位置・色の認識

        image: ndarrayで元となる画像を指定
        hint: 認識時に使用するHint情報
        result: 盤の範囲の認識結果(頂点情報等を使用する)
        戻り値: 認識成否(bool)と、成功した場合は結果情報(Result)
        """
        # 外部から直接呼ばれた場合はカメラ座標を計算する
        ret, result = self._setCameraInfo(result, hint, result.vertex, image.shape, True)
        if ret:
            # 成功した場合は石の検出を実行
            return self._detectDisc(image, hint, result)
        else:
            # force = Trueなので失敗しないはず。
            return False, None

    def extractBoard(self, image, vertex, size, ratio = 1.0, margin = 0, outer=(0, 0, 0)):
        """
        検出した盤を正方形に変換した画像を取得

        image: ndarrayで元となる画像を指定
        vertex: 盤の範囲の認識結果(頂点情報)
        size: 変換後の画像サイズ
        ratio: 頂点位置の補正用(周りの枠部分を少し拡大する用途を想定)
        margin: 変換後の画像のマージン。sizeの内数
        outer: 外側の色
        戻り値: 変換した画像(ndarray)
        """
        height = size[1]
        width = size[0]
        # 変換元の各頂点
        src = np.array(expand(vertex, ratio), dtype=np.float32)
        # 変換後の各頂点
        dst = np.array([
            [margin, margin],
            [width - 1 - margin, margin],
            [width - 1 - margin, height - 1 - margin],
            [margin, height - 1 - margin]
        ], dtype=np.float32)
        # 変換行列
        trans = cv2.getPerspectiveTransform(src, dst)
        # 変換
        board = cv2.warpPerspective(image, trans, (int(width), int(height)), \
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=outer)
        # marginを塗りつぶす
        if margin > 0:
            cv2.rectangle(board, (0, 0), (width - 1, margin), outer, -1)
            cv2.rectangle(board, (0, 0), (margin, height - 1), outer, -1)
            cv2.rectangle(board, (width - margin, 0), (width - 1, height - 1), outer, -1)
            cv2.rectangle(board, (0, height - margin), (width - 1, height - 1), outer, -1)
        return board
    
    def _detectConvexHull(self, image):
        """
        盤の範囲候補の凸包を取得するための内部関数

        image: ndarrayで元となる画像を指定
        戻り値: 認識成否(bool)と、凸包情報(ndarray)
        """
        return False, None

    def _resultForDetectBoard(self, size, hint, hull):
        """
        盤の範囲認識処理で、凸包取得後に結果を設定するための内部関数
        盤の頂点情報やカメラ位置の情報等を設定する

        size: 画像サイズ(width, height)
        hint: 認識時に使用するHint情報
        hull: _detectConvexHullで取得した凸包
        戻り値: 認識成否(bool)と、盤の認識結果(Result)
        """
        height, width, _ = size

        # 細かい凹凸を削除。盤面の線の太さ程度の凹凸は無視することにする。
        # 盤の見た目の長辺:短辺の比が極端だったとして、周長の1/2の0.625%程度の0.3%を無視するものとする。
        epsilon = self._EPSILON_COEFF * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # 得られた多角形は角が削れている可能性があるので、長い線分のトップ4を4辺とみなす
        count = len(approx)
        distances = []
        for k in range(0, count):
            distances.append(np.linalg.norm(approx[k][0] - approx[(k + 1) % count][0]))
        # 長い順にソート
        distances.sort()

        # 4位以上の線分を抽出。
        lines = []
        for k in range(0, count):
            if np.linalg.norm(approx[k][0] - approx[(k + 1) % count][0]) >= distances[count - 4] :
                lines.append([approx[k][0], approx[(k + 1) % count][0]])

        # 最悪、同点4位があるかもしれないので、4つになるまで同点4位のものを削除する
        if len(lines) > 4:
            for k in reversed(range(0, len(lines))):
                if np.linalg.norm(approx[k][0] - approx[(k + 1) % count][0]) == distances[count - 4] :
                    lines.remove(lines[k])
                    if len(lines) <= 4:
                        break

        # 4辺の交点を頂点とする
        vtx = []
        for k in range(0,4):
            vtx.append(intersection(lines[k], lines[(k + 1) % 4]))

        # 上の点から順番に並べる
        vtx = sorted(vtx, key=lambda pt: pt[1])
        # 左上の方から時計回りになるように、上2点、下2点でそれぞれ並べ変える
        if vtx[0][0] > vtx[1][0]:
            vtx = [vtx[1], vtx[0], vtx[2], vtx[3]]
        if vtx[2][0] < vtx[3][0]:
            vtx = [vtx[0], vtx[1], vtx[3], vtx[2]]

        # 面積が画像の短辺を一辺とする正方形より大きい場合は、盤がはみ出していると判断して対象外とする
        S = abs(float(np.cross(vtx[1] - vtx[0], vtx[2] - vtx[0])))
        if S >= min(width, height) ** 2:
            return False, None

        # 隣接する2点が画像の端近辺の場合ははみ出しているとみなす
        limit = 2.0
        for k in range(0, 4):
            x0, y0 = vtx[k]
            x1, y1 = vtx[(k + 1) % 4]
            if ( abs(x0) <= limit and abs(x1) <= limit ) \
                or ( abs(y0) <= limit and abs(y1) <= limit ) \
                or ( abs(x0 - width) <= limit and abs(x1 - width) <= limit ) \
                or ( abs(y0 - height) <= limit and abs(y1 - height) <= limit ):
                return False, None

        result = Result()
        ret, result = self._setCameraInfo(result, hint, vtx, size, False)
        if ret:
            # 盤の枠線を考慮して少しだけサイズを広げる
            vtx = self._adjustVertexes(vtx)
            result.vertex = vtx
            return ret, result
        return False, None

    def _adjustVertexes(self, vtx):
        """
        求めた頂点の調整
        基底クラスでは何もしない。
        """
        return vtx

    def _setCameraInfo(self, result, hint, vtx, img_size, force):
        """
        カメラの座標を計算し結果に設定する

        result: 設定対象のResultインスタンス
        hint: 認識時に使用するHint情報
        vtx: 写真上の4点(2次元座標)
        img_size: 写真のサイズ(px)
        force: 盤のチェックが失敗しても強制的に続行するかどうか
        """
        # 対角線の交点を求める
        center = intersection([vtx[0],vtx[2]], [vtx[1],vtx[3]])

        # 写真の4頂点に射影された元が正方形かどうかの判定
        # 画角が与えられていない場合は画角を求める
        if hint.focal is None:
            # まずはlog(focal) = 0と仮定して、4頂点の原像が平行四辺形だった場合の短辺長辺の比と、
            # その間の角度、3次元上の座標を求める
            focal_log = 0.0
            img_center = np.array([img_size[0] / 2, img_size[1] / 2]) # 画像の中心
            ratio, rad, vtx3d = getParallelogramRatio(vtx, center, img_size, math.exp(focal_log), img_center)
            # 正方形であればratio = 1.0, rad = pi/2 になるはずなので、誤差を求めておく
            error = (ratio - 1.0) ** 2 + (rad / (math.pi / 2) - 1.0) ** 2
            scale = 1.0
            # 誤差が小さくなるように、logの値を範囲を狭めながら近づけていく
            for i in range(0, 5):
                # 現在の最善の値
                f_log_loop_best = focal_log
                # 現在の最善のfocal_logが例えば1.4であれば、1.31,...,1.49まで試して一番良いものを選ぶ
                for j in range(-9, 10):
                    cur_focal = math.exp(focal_log + j * scale)
                    cur_ratio, cur_rad, cur_vtx3d = getParallelogramRatio(vtx, center, img_size, cur_focal, img_center)
                    cur_error = (cur_ratio - 1.0) ** 2 + (cur_rad / (math.pi / 2) - 1.0) ** 2
                    if cur_error < error:
                        error = cur_error
                        vtx3d = cur_vtx3d
                        f_log_loop_best = focal_log + j * scale
                focal_log = f_log_loop_best
                # 1桁細かい部分で再実行
                scale *= 0.1
        else:
            # ヒントとしてfocalが与えられている場合は、その値を元に誤差を算出
            ratio, rad, vtx3d = getParallelogramRatio(vtx, center, img_size, hint.focal, hint.center)
            error = (ratio - 1.0) ** 2 + (rad / (math.pi / 2) - 1.0) ** 2
        
        # 正方形っぽくなかったら失敗とする
        if error >= 0.002 and force == False :
            return False, None

        # 盤をwarpPerspectiveで変換した後のカメラ位置を求める
        # 座標系は盤の左上隅(vtx[0])からBORAD_MARGIN分拡張した点を原点とし、
        # 盤のカメラ側をz軸の負の方向とする

        # まずマージン補正前のx軸・y軸方向となるベクトルを3次元座標上で求める
        xVec = vtx3d[1] - vtx3d[0] # vtx3d[1]は右上の頂点
        yVec = vtx3d[3] - vtx3d[0] # vtx3d[3]は左下の頂点
        # 各ベクトルの長さを求めておく
        xLen = np.linalg.norm(xVec)
        yLen = np.linalg.norm(yVec)
        # 法線ベクトルを求める(z軸)
        zVec = np.cross(xVec, yVec)
        # zVecの長さを標準化しておく
        zVec = zVec / np.linalg.norm(zVec)

        # 原点(カメラ位置)と、盤面のpx距離を求める
        cameraZ = np.dot(vtx3d[0], zVec)
        # 原点(カメラ位置)から盤面に下ろした法線の足は、 cameraZ * zVec
        # その盤面上のx座標、y座標を求める(px単位)
        cameraX = np.dot(cameraZ * zVec - vtx3d[0], xVec) / xLen
        cameraY = np.dot(cameraZ * zVec - vtx3d[0], yVec) / yLen

        # マージン、縮尺を補正してカメラ位置を記憶しておく
        # 縮尺は一辺が_CELL_SIZE * 8になるようにする
        cameraPosAdjustedX = (cameraX / xLen) * BaseRecognizer._CELL_SIZE * 8 + BaseRecognizer._BOARD_MARGIN
        cameraPosAdjustedY = (cameraY / yLen) * BaseRecognizer._CELL_SIZE * 8 + BaseRecognizer._BOARD_MARGIN
        result.cameraPosition_px = np.array([cameraPosAdjustedY, cameraPosAdjustedX], dtype=np.float32)

        # 盤の中心を原点とし、盤の一辺を1とする座標系でカメラ位置を求める
        cameraPosNormalizedX = cameraX / xLen - 0.5
        cameraPosNormalizedY = cameraY / yLen - 0.5
        boardSize = (xLen + yLen) / 2
        cameraPosNormalizedZ = cameraZ / boardSize
        result.cameraPosition_bd = np.array([cameraPosNormalizedX, cameraPosNormalizedY, cameraPosNormalizedZ], dtype=np.float32)

        # 頂点の情報を設定
        result.vertex = vtx
        result.vertex3d = list(map(lambda v: v / boardSize, vtx3d))

        return True, result

    def _detectDisc(self, image, hint, result):
        """
        石の位置・色の認識を実行するための内部関数

        image: ndarrayで元となる画像を指定
        hint: 認識時に使用するHint情報
        result: 盤の範囲の認識結果(頂点情報等を使用する)
        戻り値: 認識成否(bool)と、成功した場合は結果情報(Result)
        """
        return False, None

class RealBoardRecognizer(BaseRecognizer):
    """
    実際の盤を認識するためのクラス
    """
    # 盤面抽出時の近似の係数
    # 盤の見た目の長辺:短辺の比が極端だったとして、周長の1/2の0.7%程度の0.4%を無視するものとする。
    _EPSILON_COEFF = 0.004

    # 石の色を決定する際に色を収集する半径
    _RADIUS_FOR_DISC_COLOR = 10
    # 石の色を収集するための円形のフィルタ
    _CIRCLE_FILTER = np.zeros((_RADIUS_FOR_DISC_COLOR * 2 + 1, _RADIUS_FOR_DISC_COLOR * 2 + 1), dtype=np.int8)
    _CIRCLE_FILTER = cv2.circle(_CIRCLE_FILTER, (_RADIUS_FOR_DISC_COLOR, _RADIUS_FOR_DISC_COLOR), \
        _RADIUS_FOR_DISC_COLOR, (1), -1)

    def _detectConvexHull(self, image):
        """
        盤の範囲候補の凸包を取得するための内部関数の実装
        """

        # サイズが大きい場合は計算時間短縮のため一度縮小する
        height, width, _ = image.shape
        ratio = 1.0
        if max(width, height) > 1024:
            ratio = 1024.0 / max(width, height)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
            height, width, _ = image.shape

        # 盤面抽出処理

        # ぼかし処理の後、HSV形式に変換
        hsv = cv2.cvtColor(cv2.blur(image,(3,3)), cv2.COLOR_BGR2HSV)

        # 緑色っぽい範囲を抽出
        green = cv2.inRange(hsv, 
            np.array([45, int(256*0.35), int(256*0.12)]), 
            np.array([90, 255, 255]))
        green = cv2.bitwise_or(green, 
                    cv2.inRange(hsv, 
                        np.array([45, int(256*0.25), int(256*0.35)]), 
                        np.array([90, 255, 255])))

        # 盤面の線を消す。線の太さは1マスの5%程度。
        # 画像の長辺一杯が盤面だとして、その5% * 1/8 = 0.625%程度が線の最大の太さ
        # 両サイドから膨張させて線を消したいので、その半分程度の0.35%膨張させることにする。
        kernelSize = max(1, int(0.0035 * max(width, height))) * 2 + 1
        kernel = np.ones((kernelSize, kernelSize), dtype=np.int)
        green = cv2.dilate(green, kernel)
        green = cv2.erode(green, kernel)
        
        # マス同士が石で分断されないよう、白い領域も加える
        # 黒を加えると、盤の縁経由でかなり広がってしまう危険があるので、白のみにする
        # 分断するのは斜めから見ている時なので、白も見えているはず
        greenWhite = cv2.bitwise_or(green, 
                    cv2.inRange(hsv, np.array([0, 0, 128]), np.array([180, 50, 255])))

        # 領域の輪郭を抽出
        image, contours, hierarchy = cv2.findContours(greenWhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 画像の中央を含む領域を候補として抽出する(画像中央付近に盤がある前提)
        for i, c in enumerate(contours):
            # 凸包を取得
            c = cv2.convexHull(c)
            if cv2.pointPolygonTest(c,(width / 2 , height / 2), False) > 0:
                # 領域を取得する
                mask = np.zeros((height, width),dtype=np.uint8)
                cv2.fillPoly(mask, pts=[c], color=(255))
                break
        
        if mask is None:
            # 取得失敗
            return False, None

        # 求めたhullは緑色領域に一度白色領域を加えているため、より正確に領域を求めるため、
        # 求めた範囲内で緑色領域のみのhullを取り直す

        # この領域内での膨張前の緑色領域を取得する
        green = cv2.bitwise_and(green, green, mask=mask)
        # 凸包を再度取得する
        image, contours, hierarchy = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.convexHull(functools.reduce(lambda x, y: np.append(x, y, axis=0), contours))

        # サイズを縮小していた場合は、元のサイズに戻す
        if ratio > 0:
            c = c / ratio
        return True, c.astype(np.float32)

    def _adjustVertexes(self, vtx):
        """
        求めた頂点の調整
        実盤だと緑色の部分の範囲を盤の範囲として認識してしまうため、周りの枠線を含めるように
        少しだけ広げる。
        """
        return expand(vtx, self._CELL_SIZE * 4 / (self._CELL_SIZE * 4 - 1) )

    def _detectDisc(self, image, hint, result):
        """
        石の位置・色の認識を実行するための内部関数
        """
        # 結果の石情報のクリア
        result.clearDiscInfo()

        # 盤を切り出した画像の取得
        board = self._extractBoardForDetectDisc(image, result)

        # 盤の外部(枠外)を抽出
        outer = cv2.inRange(board, (254, 0, 0), (254, 0, 0))

        # 盤面をHSVに変換
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)

        # 緑色の部分(盤とみなせる部分)を抽出
        green = self._extractGreenForDetectDisc(hsv, outer, hint)

        # 緑色以外の領域(≒石の領域)
        notGreen = cv2.bitwise_not(green)

        # 色のついている(白黒以外の)領域(Sの値が小さい)
        colored = cv2.inRange(hsv, np.array([0, 127, 30]), np.array([180, 255, 255]))
        colored = cv2.bitwise_or(colored, cv2.inRange(hsv, np.array([0, 100, 50]), np.array([180, 255, 255])))
        colored = cv2.bitwise_or(colored, cv2.inRange(hsv, np.array([0, 75, 150]), np.array([180, 255, 255])))
        colored = cv2.bitwise_or(colored, cv2.inRange(hsv, np.array([0, 50, 200]), np.array([180, 255, 255])))

        # グレースケールの盤の画像
        grayBoard = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

        # 石の認識に必要な情報の取得
        info = self._prepareInfoForDetectDisc(grayBoard)

        # 着色領域の処理と石の領域の取得
        disc, info, result = self._processColoredAndExtractDiscForDetectDisc(notGreen, colored, outer, hint, info, result)

        # 石の外側からの距離を求める
        dist = cv2.distanceTransform(disc, cv2.DIST_L2, 5)
        # 外側からの距離が13以上のブロックに分ける。これによって隣接している石が分離できる
        _, distThr = cv2.threshold(dist, 13.0, 255.0, cv2.THRESH_BINARY)
        distThr = np.uint8(distThr) # こうしておかないとconnectedComponentsが通らない

        # 各連結成分が石に対応するので1つずつ判別していく
        labelnum, labelimg, data, center = cv2.connectedComponentsWithStats(distThr)
        for i in range(1, labelnum):
            x, y, w, h, s = data[i]
            if s > 2500:
                # 大きすぎる場合は何か違うものが写っているとみなして、不明マスと扱う
                result = self._setColorUnknown(labelimg, x, y, w, h, i, result)
                continue
            
            # 該当箇所を切り出す
            distComponent = dist[y:y+h, x:x+w]
            # 普通は連結領域1つに1つの石だが、双峰っぽくなっている場合もあるのでループして消しこみながら判定する
            while True:
                # 領域内の距離の最大値とその場所を求める
                maxCoord = argmax(distComponent)
                maxVal = distComponent[maxCoord]

                if maxVal < 13.0:
                    # 検出を終えているのでループを抜ける
                    break
                elif maxVal >= 20.0 and hint.mode == Mode.VIDEO:
                    # VIDEOモードで大きすぎる場合は手などが写っているとみなして、不明マスと扱う
                    result = self._setColorUnknown(labelimg, x, y, w, h, i, result)

                # 石の色の判定
                result = self._detectDiscColor(distComponent, x, y, info, result, maxCoord)

                # 判定済みの箇所を消し込む
                cv2.circle(distComponent, (maxCoord[1], maxCoord[0]), int(maxVal * 1.4), 0, -1)

        return True, result
    
    def _extractBoardForDetectDisc(self, image, result):
        """
        盤面画像を抽出するための内部関数(石の位置・色の認識用)
        """
        # 実盤の場合は外周の枠線が少し太いので、範囲を少し広げて画像を取得
        # outerはあとで抽出しやすいようにほぼ出て来なさそうな色を指定しておく
        return self.extractBoard(image, result.vertex, \
            [BaseRecognizer._EXTRACT_IMG_SIZE, BaseRecognizer._EXTRACT_IMG_SIZE], \
            ratio=(self._CELL_SIZE * 4 + 2) / (self._CELL_SIZE * 4), \
            margin=BaseRecognizer._BOARD_MARGIN - 2, outer=(254, 0, 0))

    def _extractGreenForDetectDisc(self, hsv, outer, hint):
        """
        緑色部分(石ではない部分)を抽出するための内部関数(石の位置・色の認識用)
        """
        # 緑色領域を取り出し
        green = cv2.inRange(hsv, np.array([45, int(255 * 0.35), int(255 * 0.12)]), np.array([90, 255, 255]))
        green = cv2.bitwise_or(green, \
            cv2.inRange(hsv, np.array([45, int(255 * 0.25), int(255 * 0.35)]), np.array([90, 255, 255])))

        # 外周部分は緑とみなす
        green = cv2.bitwise_or(green, outer)
        return green

    def _prepareInfoForDetectDisc(self, grayBoard):
        """
        石の認識に必要な各種情報を準備するための内部関数(石の位置・色の認識用)
        """
        # 二値化
        # 見た目の色を判断するための広域的な二値化
        binBoardWide = cv2.adaptiveThreshold(grayBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -20)
        # 反射した石は少しまだらになるので、それが検出できるように近傍で二値化したものも作っておく
        binBoardNarrow = cv2.adaptiveThreshold(grayBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        binBoardNarrow = cv2.blur(binBoardNarrow, (3, 3))
        _, binBoardNarrow = cv2.threshold(binBoardNarrow, 168, 255, cv2.THRESH_BINARY)

        return { "wide": binBoardWide, "narrow": binBoardNarrow}

    def _processColoredAndExtractDiscForDetectDisc(self, notGreen, colored, outer, hint, info, result):
        """
        着色領域の処理と石の領域を取得するための内部関数(石の位置・色の認識用)
        """
        if hint.mode == Mode.VIDEO:
            # 緑色以外のある程度の大きさの着色領域は、手などが写っているとみなして、不明マスと扱う
            colored = cv2.bitwise_and(colored, notGreen)
            outer = cv2.bitwise_or(colored, outer)

            # 不明マスの設定
            labelnum, labelimg, data, center = cv2.connectedComponentsWithStats(outer)
            for i in range(1, labelnum):
                x, y, w, h, s = data[i]
                if s >= 200:
                    # 面積が200以上であれば不明マス扱いとする
                    result = self._setColorUnknown(labelimg, x, y, w, h, i, result)
        return notGreen, info, result

    def _detectDiscColor(self, distComponent, x, y, info, result, maxCoord):
        """
        石の色を判断する
        """
        # カメラの位置から石の位置に向けた方向を求める
        direction = maxCoord + np.array([y, x]) - result.cameraPosition_px

        # maxCoord付近の最大値に近い点の内、この向きに対して遠い方が石の表面の中心、近い方が石の底面の中心
        maxEdgePoint, minEdgePoint = getRidgeEdge(distComponent, maxCoord, direction)

        # 色を判定する範囲
        startX = maxEdgePoint[1] + x - RealBoardRecognizer._RADIUS_FOR_DISC_COLOR
        startY = maxEdgePoint[0] + y - RealBoardRecognizer._RADIUS_FOR_DISC_COLOR
        endX = maxEdgePoint[1] + x + RealBoardRecognizer._RADIUS_FOR_DISC_COLOR + 1
        endY = maxEdgePoint[0] + y + RealBoardRecognizer._RADIUS_FOR_DISC_COLOR + 1

        # maxEdgePointを中心とする円内で、binBoardWideで黒の面積を求める
        subBinWide = info["wide"][startY:endY, startX:endX]
        circleWide = subBinWide[RealBoardRecognizer._CIRCLE_FILTER == 1]
        areaWide = circleWide[circleWide == 0].shape[0]

        # maxEdgePointを中心とする円内で、binBoardNarrowで黒の面積を求める
        subBinNarrow = info["narrow"][startY:endY, startX:endX]
        circleNarrow = subBinNarrow[RealBoardRecognizer._CIRCLE_FILTER == 1]
        areaNarrow = circleNarrow[circleNarrow == 0].shape[0]

        # 底面の座標
        bottomCoord = minEdgePoint + np.array([y, x]) - np.array([RealBoardRecognizer._BOARD_MARGIN, RealBoardRecognizer._BOARD_MARGIN])
        bottomCoord = bottomCoord / (RealBoardRecognizer._CELL_SIZE * 8)
        # セルの位置
        bottomIndex = np.array([0, 0])
        bottomIndex[0] = min(7, max(0, int(bottomCoord[0] / 0.125)))
        bottomIndex[1] = min(7, max(0, int(bottomCoord[1] / 0.125)))

        # 色の判定
        if areaWide >= 10:
            # 黒
            self._setDisc(result, DiscColor.BLACK, bottomCoord, bottomIndex)
        elif areaNarrow >= 26:
            # 黒
            self._setDisc(result, DiscColor.BLACK, bottomCoord, bottomIndex)
        else:
            # 白
            self._setDisc(result, DiscColor.WHITE, bottomCoord, bottomIndex)
        
        return result

    def _setDisc(self, result, color, coord, index):
        """
        結果に石情報を設定
        """
        if result.isUnknown[tuple(index)]:
            # 不明マスの場合は無視する
            return result
        
        # 石情報を追加
        disc = Disc()
        disc.color = color
        disc.position = coord
        disc.cell = index
        result.disc.append(disc)
        return result

    def _setColorUnknown(self, labelimg, x, y, w, h, idx, result):
        """
        labelimgの中でidxの値が入っているマスをUnknown扱いとする
        チェック範囲はx〜x+w, y〜y+hとする
        """
        # 盤のマスごとにチェックする
        startI = max(0, int((x - BaseRecognizer._BOARD_MARGIN) / BaseRecognizer._CELL_SIZE))
        startJ = max(0, int((y - BaseRecognizer._BOARD_MARGIN) / BaseRecognizer._CELL_SIZE))
        endI = min(7, int((x + w - BaseRecognizer._BOARD_MARGIN) / BaseRecognizer._CELL_SIZE))
        endJ = min(7, int((y + h - BaseRecognizer._BOARD_MARGIN) / BaseRecognizer._CELL_SIZE))

        for i in range(startI, endI + 1):
            topX = i * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN
            bottomX = (i + 1) * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN
            for j in range(startJ, endJ + 1):
                # 既にUnknownの場合は判定しない
                if result.isUnknown[j, i]:
                    continue
                topY = j * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN
                bottomY = (j + 1) * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN
                # セルの範囲
                cell = labelimg[topY:bottomY, topX:bottomX]
                # その範囲内にidxの値が存在する場合はUnknownとする
                if len(cell[cell == idx]) > 0:
                    result.isUnknown[j, i] = True
        return result

class ScreenshotRecognizer(RealBoardRecognizer):
    """
    スマホアプリのスクリーンショット等の盤を認識するためのクラス
    mode == PHOTOのみ対応する
    """
    # オセロクエスト対策用に使用するフィルタ。各マスの中央のマスクを行う
    _OQ_MASK = np.zeros((RealBoardRecognizer._EXTRACT_IMG_SIZE, RealBoardRecognizer._EXTRACT_IMG_SIZE), dtype=np.uint8)
    _RADIUS_FOR_OQ_MASK = 14

    # マスの石以外の部分の色が特殊(白黒緑以外)な場合に無視するためのフィルタ。各マスの中央以外の部分のマスクを行う
    _COLORED_MASK = np.ones((RealBoardRecognizer._EXTRACT_IMG_SIZE, RealBoardRecognizer._EXTRACT_IMG_SIZE) \
        , dtype=np.uint8) * 255
    _RADIUS_FOR_COLORED_MASK = 11

    def _detectConvexHull(self, image):
        """
        盤の範囲候補の凸包を取得するための内部関数の実装
        """
        # Twitterの背景色対策。Twitterアプリ上の画像表示だと画像の周囲に緑の背景が表示され、
        # 誤認識の元になるので、マスクする。
        # 上辺と右辺のどちらかが単色で緑色っぽい場合にマスクする。
        height, width, _ = image.shape
        up = image[0, :, :]
        right = image[:, width - 1, :]
        outer = np.zeros((height, width), dtype=np.uint8)
        for matrix in [up, right]:
            minB = min(matrix[:, 0])
            maxB = max(matrix[:, 0])
            minG = min(matrix[:, 1])
            maxG = max(matrix[:, 1])
            minR = min(matrix[:, 2])
            maxR = max(matrix[:, 2])
            # 単色のチェック
            if minB == maxB and minG == maxG and minR == maxR:
                # 緑っぽいかどうかのチェック。iOSのTwitterアプリでオセロ盤が写っているときの背景色はこれくらいの色
                if minG >= 60 and minG <= 68 and minB < minG and minR < minG:
                    # 背景色部分の抽出
                    bgColor = np.array([minB, minG, minR])
                    outer = cv2.bitwise_or(outer, cv2.inRange(image, bgColor, bgColor))
                    inner = cv2.bitwise_not(outer)
                    image = cv2.bitwise_and(image, image, mask=inner)
                    break

        # サイズが大きい場合は計算時間短縮のため一度縮小する
        ratio = 1.0
        if max(width, height) > 1024:
            ratio = 1024.0 / max(width, height)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
            height, width, _ = image.shape

        # 盤面抽出処理
        # 緑色領域＋水色領域(iEdax対策)を取得
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([40, int(255 * 0.5), int(255 * 0.12)]), np.array([100, 255, 255]))
        green = cv2.bitwise_or(green, cv2.inRange(hsv, np.array([40, int(255 * 0.35), int(255 * 0.25)]), np.array([100, 255, 255])))
        green = cv2.bitwise_or(green, cv2.inRange(hsv, np.array([40, int(255 * 0.25), int(255 * 0.35)]), np.array([100, 255, 255])))

        # ノイズを除去
        green = cv2.dilate(green, _KERNEL9)
        green = cv2.erode(green, _KERNEL9)

        # スクリーンショットなので、余白領域が白色になっていることが多く、白色部を挟んで隣接する別の緑領域に
        # 連結してしまわないように、白領域で分断する
        white = cv2.inRange(hsv, np.array([0, 0, int(255 * 0.97)]), np.array([180, int(255 * 0.15), 255]))
        green = cv2.bitwise_and(green, green, mask=cv2.bitwise_not(white))

        # 連結領域の内、面積が最大のものを取得する
        labelnum, labelimg, data, center = cv2.connectedComponentsWithStats(green)
        area = np.array(data)[1:, 4] # 先頭は外側部分のため除外
        maxIdx = np.argmax(area)
        maxImg = cv2.inRange(labelimg, np.array([maxIdx + 1]), np.array([maxIdx + 1])) # 先頭を省いて最大を求めたため、+1した

        # 最大領域の輪郭を取得する
        _, contours, _ = cv2.findContours(maxImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 1件でない場合は探索失敗とする
        if len(contours) != 1:
            return False, None

        epsilon = 0.03 * cv2.arcLength(contours[0], True)
        c = cv2.approxPolyDP(contours[0], epsilon, True)

        # 四角形でない場合は探索失敗とする
        if c.shape[0] != 4:
            return False, None

        # 最大の辺とbounding boxを取得する
        maxEdge = 0
        minX = width
        minY = height
        maxX = 0
        maxY = 0
        for i in range(0, 4):
            edge = c[(i + 1) % 4, 0] - c[i % 4, 0]
            maxEdge = max(maxEdge, abs(edge[0]), abs(edge[1])) # X方向かY方向の長い方
            minX = min(minX, c[i % 4, 0, 0])
            minY = min(minY, c[i % 4, 0, 1])
            maxX = max(maxX, c[i % 4, 0, 0])
            maxY = max(maxY, c[i % 4, 0, 1])

        # 4辺の長さがほぼ等しいことと、X軸・Y軸にほぼ平行であることをチェックする。
        # 違う場合はスクリーンショットではないと判断
        dLenLimit = max(int(maxEdge * 0.05), 1) # 5%まで長さの差を許容
        diffLimit = max(int(maxEdge * 0.005), 1) # 平行のずれを辺の長さの0.5%まで許容
        for i in range(0, 4):
            edge = c[(i + 1) % 4, 0] - c[i % 4, 0]
            edgeLen = max(abs(edge[0]), abs(edge[1]))
            diff = min(abs(edge[0]), abs(edge[1])) # 軸に平行でない成分
            if maxEdge - edgeLen > dLenLimit or diff > diffLimit:
                return False, None

        # 一辺が短辺の2割以上とする
        if maxEdge < min(width, height) * 0.2:
            return False, None

        # 盤の領域を切り出す
        board = image[minY:maxY, minX:maxX]

        # グレースケール化
        grayBoard = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

        # 求めたの盤は緑色領域。アプリの盤は枠線の外側に緑色のスペースがあるケースがあるため、
        # マスの線の範囲を元にX,Y方向の実際の盤の範囲を求める。
        # まずはY方向の範囲を求めるため、X方向で微分(Sobel)して範囲を決める。
        sobelH = cv2.Sobel(grayBoard, cv2.CV_32F, 1, 0, ksize=1)
        # 各yに対して、Y方向の線が存在するとみなして良さそうな領域
        lineH = cv2.inRange(sobelH, -255, -15)
        # 検出した線の本数っぽい値
        sumH = np.sum(lineH, axis=1) / 255
        y_start = 0
        for y in range(3, sumH.shape[0]): # 最初の3pixelくらいはノイズが入ることがあるので見ない
            if sumH[y] >= 3: # 線が3本以上あればこの値は3以上にはなっているはず
                y_start = y
                break
        y_start = 0 if y_start == 3 else y_start # 最初の3pxを読み飛ばしたので、3の場合は最初からあるとみなす
        minY = minY + y_start # minYの値を補正

        # 同様にmaxYを補正する
        y_end = 0
        for y in range(3, sumH.shape[0]):
            if sumH[sumH.shape[0] - y - 1] >= 3:
                y_end = y
                break
        y_end = 0 if y_end == 3 else y_end
        maxY = maxY - y_end

        # X方向についても同様に求める
        sobelV = cv2.Sobel(grayBoard, cv2.CV_32F, 0, 1, ksize=1)
        lineV = cv2.inRange(sobelV, -255, -15)
        sumV = np.sum(lineV, axis=0) / 255
        x_start = 0
        for x in range(3, sumV.shape[0]):
            if sumV[x] >= 3:
                x_start = x
                break
        x_start = 0 if x_start == 3 else x_start
        minX = minX + x_start

        x_end = 0
        for x in range(3, sumV.shape[0]):
            if sumV[sumV.shape[0] - x - 1] >= 3:
                x_end = x
                break
        x_end = 0 if x_end == 3 else x_end
        maxX = maxX - x_end

        # 正方形に近いか再確認する
        if maxX - minX < (maxY - minY) * 0.995 or maxX - minX > (maxY - minY) * 1.005:
            return False, None

        # 短辺の2割以上か確認する
        if maxX - minX < min(width, height) * 0.2:
            return False, None

        hull = np.array([[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY]], dtype=np.float32)
        # サイズを縮小していた場合は、元のサイズに戻す
        if ratio > 0:
            hull = hull / ratio
        return True, hull

    def _adjustVertexes(self, vtx):
        """
        求めた頂点の調整
        スクリーンショットの場合はそのままとする
        """
        return vtx

    def _setCameraInfo(self, result, hint, vtx, img_size, force):
        """
        カメラの座標を計算し結果に設定する
        スクリーンショットの場合は、頂点情報の設定のみ行う
        (正方形であることは確認済み、カメラ情報は使用しないため)
        """
        result.vertex = vtx
        return True, result

    def _extractBoardForDetectDisc(self, image, result):
        """
        盤面画像を抽出するための内部関数(石の位置・色の認識用)
        """
        # スクリーンショットの場合はそのまま抽出
        # outerはあとで抽出しやすいようにほぼ出て来なさそうな色を指定しておく
        return self.extractBoard(image, result.vertex, \
            [BaseRecognizer._EXTRACT_IMG_SIZE, BaseRecognizer._EXTRACT_IMG_SIZE], \
            ratio=1.0, \
            margin=BaseRecognizer._BOARD_MARGIN, outer=(254, 0, 0))

    def _extractGreenForDetectDisc(self, hsv, outer, hint):
        """
        緑色部分(石ではない部分)を抽出するための内部関数(石の位置・色の認識用)
        """
        green = super()._extractGreenForDetectDisc(hsv, outer, hint)

        # iEdax対策として、写真の場合は水色の部分も追加
        if hint.mode == Mode.PHOTO:
            green = cv2.bitwise_or(green, \
                cv2.inRange(hsv, np.array([45, int(255 * 0.25), int(255 * 0.75)]), np.array([100, 255, 255])))

        return green

    def _prepareInfoForDetectDisc(self, grayBoard):
        """
        石の認識に必要な各種情報を準備するための内部関数(石の位置・色の認識用)
        """
        # 二値化
        # 見た目の色を判断するための広域的な二値化
        binBoardWide = cv2.adaptiveThreshold(grayBoard, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -20)
        # スクリーンショットは石が反射しないので、近傍の二値化は不要(意匠性の高い石の画像だとかえって精度が悪化する)

        return { "wide": binBoardWide }

    def _processColoredAndExtractDiscForDetectDisc(self, notGreen, colored, outer, hint, info, result):
        """
        着色領域の処理と石の領域を取得するための内部関数(石の位置・色の認識用)
        """
        # 各マスの中央付近以外にある着色領域は緑色と同じ扱いにする。不明マス扱いにはしない。
        # (最終手のマス全体が着色されているようなアプリの対応)
        coloredOut = cv2.bitwise_and(colored, ScreenshotRecognizer._COLORED_MASK)
        notColoredOut = cv2.bitwise_not(coloredOut)
        disc = cv2.bitwise_and(notGreen, notColoredOut)

        # 着色領域を二値化画像(0が黒)にorで加算して白っぽくしておく
        # (最終手の石の中央に赤四角等が表示されているようなアプリで、黒石と判定しないための対処)
        colored = cv2.dilate(colored, _KERNEL5)
        binBoardWide = cv2.bitwise_or(info["wide"], colored)
        info["wide"] = binBoardWide

        # オセロクエストの旧スタイルの黒石に緑色っぽい画素が含まれているのでその対策
        # 各マスの中央付近はノイズを除去しておく
        # 石の狭間でわずかに見えている緑色を消してしまわないよう、マスの中央付近以外はノイズ除去対象外とする
        discDenoised = cv2.dilate(disc, _KERNEL5)
        discDenoised = cv2.erode(discDenoised, _KERNEL5)
        discDenoised = cv2.bitwise_and(discDenoised, discDenoised, mask=ScreenshotRecognizer._OQ_MASK)
        disc = cv2.bitwise_or(disc, discDenoised)

        return disc, info, result

    def _detectDiscColor(self, distComponent, x, y, info, result, maxCoord):
        """
        石の色を判断する
        """
        # 色を判定する範囲
        startX = maxCoord[1] + x - RealBoardRecognizer._RADIUS_FOR_DISC_COLOR
        startY = maxCoord[0] + y - RealBoardRecognizer._RADIUS_FOR_DISC_COLOR
        endX = maxCoord[1] + x + RealBoardRecognizer._RADIUS_FOR_DISC_COLOR + 1
        endY = maxCoord[0] + y + RealBoardRecognizer._RADIUS_FOR_DISC_COLOR + 1

        # maxCoordを中心とする円内で、binBoardWideで黒の面積を求める
        subBinWide = info["wide"][startY:endY, startX:endX]
        circleWide = subBinWide[RealBoardRecognizer._CIRCLE_FILTER == 1]
        areaWide = circleWide[circleWide == 0].shape[0]

        # 中心の座標
        discCoord = maxCoord + np.array([y, x]) - np.array([RealBoardRecognizer._BOARD_MARGIN, RealBoardRecognizer._BOARD_MARGIN])
        discCoord = discCoord / (RealBoardRecognizer._CELL_SIZE * 8)
        # セルの位置
        discIndex = np.array([0, 0])
        discIndex[0] = min(7, max(0, int(discCoord[0] / 0.125)))
        discIndex[1] = min(7, max(0, int(discCoord[1] / 0.125)))

        # 色の判定
        if areaWide >= 10:
            # 黒
            self._setDisc(result, DiscColor.BLACK, discCoord, discIndex)
        else:
            # 白
            self._setDisc(result, DiscColor.WHITE, discCoord, discIndex)
        
        return result


# ScreenshotRecognizerのstaticな変数の初期化
for j in range(0, 8):
    for i in range(0, 8):
        # OQ_MASK用の円の描画
        ScreenshotRecognizer._OQ_MASK = cv2.circle(ScreenshotRecognizer._OQ_MASK, \
            (int((i + 0.5) * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN), \
                int((j + 0.5) * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN)), \
            ScreenshotRecognizer._RADIUS_FOR_OQ_MASK, \
            (255), \
            -1
        )

        # COLORED_MASK用の円の描画(中央を0で塗りつぶす)
        ScreenshotRecognizer._COLORED_MASK = cv2.circle(ScreenshotRecognizer._COLORED_MASK, \
            (int((i + 0.5) * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN), \
                int((j + 0.5) * BaseRecognizer._CELL_SIZE + BaseRecognizer._BOARD_MARGIN)), \
            ScreenshotRecognizer._RADIUS_FOR_COLORED_MASK, \
            (0), \
            -1
        )


class PrintedBoardRecognizer(RealBoardRecognizer):
    """
    書籍等、印刷された白黒の盤を認識するためのクラス
    """
    # 盤面抽出時の近似の係数
    # 文字等がくっついている可能性があるので3%程度。
    _EPSILON_COEFF = 0.03

    def _detectConvexHull(self, image):
        """
        盤の範囲候補の凸包を取得するための内部関数
        """
        height, width, _ = image.shape

        # サイズが大きい場合は計算時間短縮のため一度縮小する
        ratio = 1.0
        if max(width, height) > 1024:
            ratio = 1024.0 / max(width, height)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
            height, width, _ = image.shape

        # 画像をグレースケールにする
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二値化。adaptiveThresholdのblockSizeは短辺の約1/3とする。blockSizeは奇数になるように。
        blockSize = int(math.floor((min(width, height) - 1) / 6)) * 2 + 1
        blackImage = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, 16)
        blackImage = cv2.blur(blackImage, (3,3))
        blackImage = cv2.inRange(blackImage, np.array([0]), np.array([216]))

        # 領域の輪郭を抽出
        image, contours, hierarchy = cv2.findContours(blackImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            # 4点未満の場合はスキップ
            if len(contours[i]) < 4:
                continue
            # 画像の中央を含む領域を候補として抽出する
            if cv2.pointPolygonTest(c,(width / 2 , height / 2), False) > 0:
                # サイズを縮小していた場合は、元のサイズに戻す
                if ratio > 0:
                    c = c / ratio
                return True, c.astype(np.float32)

        # 見つからなかった場合
        return False, None

    def _detectDisc(self, image, hint, result):
        """
        石の位置・色の認識を実行するための内部関数
        """
        # 結果の石情報のクリア
        result.clearDiscInfo()

        # 画像をグレースケールにする
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 盤を切り出した画像の取得
        board = self.extractBoard(gray, result.vertex, \
            [BaseRecognizer._EXTRACT_IMG_SIZE, BaseRecognizer._EXTRACT_IMG_SIZE], \
            ratio=1.0, margin=BaseRecognizer._BOARD_MARGIN, outer=(96))

        # 二値化
        binBoard = cv2.adaptiveThreshold(board, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 2)
        # 黒を強調
        binBoard = cv2.erode(binBoard, _KERNEL3)
        # 黒領域
        black = cv2.bitwise_not(binBoard)
        black = cv2.erode(black, _KERNEL3)

        # 書籍等の白黒盤面の場合、白い領域は盤なのか白石なのかそのままだとわからない。
        # そこで、盤のマス目の線の交点付近の白領域(の連結成分)は盤だとみなし、それ以外を白石とみなす
        # 交点は画像の座標から大体は分かるが、ずれると精度が落ちるので、なるべく正確に交点を求めるよう解析する

        # 黒領域の外部からの距離
        dist = cv2.distanceTransform(black, cv2.DIST_L2, 5)
        # そのラプラシアン
        lap = cv2.Laplacian(dist, cv2.CV_32F)
        # 負の部分が線っぽい領域
        edges = cv2.inRange(lap, np.array([-255.0]), np.array([-1.0]))

        GAP = 15 # 想定される交点の位置から許容するずれの範囲
        STAT_RANGE = 18 # 交点を求めるために統計を取る範囲の幅
        BOARD_MARGIN = BaseRecognizer._BOARD_MARGIN
        CELL_SIZE = BaseRecognizer._CELL_SIZE
        IMAGE_SIZE = BaseRecognizer._EXTRACT_IMG_SIZE
        corner = np.array([[[0, 0]] * 9 for i in range(9)])
        # まずは内部の交点(外枠以外)について、交点を判定する
        for j in range(1, 8):
            for i in range(1, 8):
                # 交点のX座標調査用の調査範囲
                minX = max(BOARD_MARGIN + i * CELL_SIZE - GAP, 0)
                maxX = min(BOARD_MARGIN + i * CELL_SIZE + GAP, IMAGE_SIZE)
                minY = max(BOARD_MARGIN + j * CELL_SIZE - STAT_RANGE, BOARD_MARGIN)
                maxY = min(BOARD_MARGIN + j * CELL_SIZE + STAT_RANGE, IMAGE_SIZE - BOARD_MARGIN)
                cornerEdges = edges[minY : maxY, minX : maxX]
                # その範囲で、線っぽい領域が最も多いX座標が交点のX座標とする
                cornerX = int(np.argmax(np.sum(cornerEdges, axis=0))) + minX

                # 同様にY座標についても求める
                minX = max(BOARD_MARGIN + i * CELL_SIZE - STAT_RANGE, BOARD_MARGIN)
                maxX = min(BOARD_MARGIN + i * CELL_SIZE + STAT_RANGE, IMAGE_SIZE - BOARD_MARGIN)
                minY = max(BOARD_MARGIN + j * CELL_SIZE - GAP, 0)
                maxY = min(BOARD_MARGIN + j * CELL_SIZE + GAP, IMAGE_SIZE)
                cornerEdges = edges[minY : maxY, minX : maxX]
                cornerY = int(np.argmax(np.sum(cornerEdges, axis=1))) + minY

                corner[i, j] = np.array([cornerY, cornerX])

        # 外枠の交点については周囲の様子によってずれやすいので、内部の点から推測する
        # 隅の四点
        corner[0, 0] = corner[1, 1] + (corner[1, 1] - corner[2, 2])
        corner[0, 8] = corner[1, 7] + (corner[1, 7] - corner[2, 6])
        corner[8, 0] = corner[7, 1] + (corner[7, 1] - corner[6, 2])
        corner[8, 8] = corner[7, 7] + (corner[7, 7] - corner[6, 6])
        # それ以外の外枠の交点
        for i in range(1, 8):
            corner[0, i] = corner[1, i] + (corner[1, i] - corner[2, i])
            corner[8, i] = corner[7, i] + (corner[7, i] - corner[6, i])
            corner[i, 0] = corner[i, 1] + (corner[i, 1] - corner[i, 2])
            corner[i, 8] = corner[i, 7] + (corner[i, 7] - corner[i, 6])

        # 求めた交点の座標付近のマスク画像を作成
        cornerMask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        MASK_SIZE = 11 # マスクの大きさ
        for j in range(0, 9):
            for i in range(0, 9):
                pt = np.array([
                    [corner[i, j, 1] - MASK_SIZE, corner[i, j, 0]],
                    [corner[i, j, 1], corner[i, j, 0] - MASK_SIZE],
                    [corner[i, j, 1] + MASK_SIZE, corner[i, j, 0]],
                    [corner[i, j, 1], corner[i, j, 0] + MASK_SIZE],
                ])
                cv2.fillConvexPoly(cornerMask, pt, 255)

        # 石の外部(盤の部分)を取得する
        labelnum, labelimg, data, center = cv2.connectedComponentsWithStats(binBoard, connectivity=8, ltype=cv2.CV_16U)

        # 外部とみなすのは、cornerMaskの領域なので、そのindexの集合を取得する
        outIdx = np.unique(labelimg[cornerMask != 0])
        # 外部領域用
        outer = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        # 各マスの内部の白色領域の面積取得用
        areaIn = np.zeros((8, 8), dtype=np.uint32)
        for i in range(1, labelnum):
            x, y, w, h, s = data[i]
            if np.in1d( i, outIdx ):
                # 外部領域のindexリストにあれば、外部領域にマージ
                outer = cv2.bitwise_or(outer, cv2.inRange(labelimg, i, i))
            elif ( x >= BOARD_MARGIN and x < IMAGE_SIZE - BOARD_MARGIN and y >= BOARD_MARGIN and y < IMAGE_SIZE - BOARD_MARGIN ):
                # それ以外は内部の石の領域と思われるため、面積をマス毎に加算する
                cell_i = int((x - BOARD_MARGIN) / CELL_SIZE)
                cell_j = int((y - BOARD_MARGIN) / CELL_SIZE)
                if ( cell_i >= 0 and cell_i < 8 and cell_j >= 0 and cell_j < 8 ):
                    areaIn[cell_j, cell_i] += s
        
        # 内部領域(外部以外の領域)。これで黒石も白石も塗りつぶされる
        inner = cv2.bitwise_not(outer)
        # その外からの距離を取得
        dist = cv2.distanceTransform(inner, cv2.DIST_L2, 5)

        # 石の色の判定
        for j in range(0, 8):
            for i in range(0, 8):
                # マスの中心の座標は、周囲の4交点の重心とする
                center = ((corner[i, j] + corner[i + 1, j + 1] + corner[i + 1, j] + corner[i, j + 1]) / 4).astype(np.int32)
                # 中心の距離の値が11.5未満の場合は石ではないとみなす(説明のための数字が書いてある等)
                if dist[center[0], center[1]] < 11.5:
                    continue
                centerPos = (center - np.array([BOARD_MARGIN, BOARD_MARGIN])) / (CELL_SIZE * 8)
                index = np.array([j, i])
                if areaIn[j, i] > 100:
                    # 内部領域が多い場合は白石
                    self._setDisc(result, DiscColor.WHITE, centerPos, index)
                else:
                    # 少ない場合は黒石
                    self._setDisc(result, DiscColor.BLACK, centerPos, index)

        return True, result

class AutomaticRecognizer(BaseRecognizer):
    """
    自動認識のためのクラス
    """
    # このクラス配下で使用するRecognizer
    _REAL_RECOGNIZER = RealBoardRecognizer()
    _SCREENSHOT_RECOGNIZER = ScreenshotRecognizer()
    _PRINTED_RECOGNIZER = PrintedBoardRecognizer()

    def detectBoard(self, image, hint):
        """
        盤の範囲の認識
        """
        if self._isColoredImage(image):
            # まずはスクリーンショットと仮定して実行
            ret, result = AutomaticRecognizer._SCREENSHOT_RECOGNIZER.detectBoard(image, hint)
            if ret:
                # 成功した場合はスクリーンショットで正解
                result.recognizerType = RecognizerType.SCREENSHOT
                return ret, result
            else:
                # 失敗した場合は実盤として実行
                ret, result = AutomaticRecognizer._REAL_RECOGNIZER.detectBoard(image, hint)
                if ret:
                    result.recognizerType = RecognizerType.REALBOARD
                    return ret, result
                else:
                    return False, None
        else:
            # 白黒盤として実行
            ret, result = AutomaticRecognizer._PRINTED_RECOGNIZER.detectBoard(image, hint)
            if ret:
                result.recognizerType = RecognizerType.PRINTED
                return ret, result
            else:
                return False, None

    def detectDisc(self, image, hint, result):
        """
        石の位置・色の認識
        """
        # 指定されてrecognizerTypeで実行
        if result.recognizerType == RecognizerType.REALBOARD:
            return AutomaticRecognizer._REAL_RECOGNIZER.detectDisc(image, hint, result)
        elif result.recognizerType == RecognizerType.SCREENSHOT:
            return AutomaticRecognizer._SCREENSHOT_RECOGNIZER.detectDisc(image, hint, result)
        elif result.recognizerType == RecognizerType.PRINTED:
            return AutomaticRecognizer._PRINTED_RECOGNIZER.detectDisc(image, hint, result)
        else:
            # recognizerType未設定の場合は失敗
            return False, None

    def _isColoredImage(self, image):
        """
        色つき画像かどうかを判定する
        """
        height, width, _ = image.shape
        # 中央付近の領域(短辺の20%)の色合いで決める
        size = min(width, height) * 0.2
        clip = image[round((height - size) / 2) : round((height + size) / 2),
            round((width - size) / 2) : round((width + size) / 2)]

        # HSVに変換
        hsv = cv2.cvtColor(clip, cv2.COLOR_BGR2HSV)

        # 着色エリアを取得(Sが大きい)
        colored = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([180, 255, 255]))
        colored = cv2.bitwise_or(colored, cv2.inRange(hsv, np.array([0, 75, 150]), np.array([180, 255, 255])))

        # 多少でも色があれば色ありと判定
        return colored[colored == 255].shape[0]/ (hsv.shape[0] * hsv.shape[1]) > 0.001; 
