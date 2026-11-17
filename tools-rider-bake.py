# 人の3Dモデル（glTF）を、競艇の艇に乗る姿勢に曲げて固め、ゲーム用の小さなデータにする
import json, struct, sys, numpy as np

G = json.load(open('male.gltf'))
BIN = open('male.bin', 'rb').read()

def acc(i):
    a = G['accessors'][i]; bv = G['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
    dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a['componentType']]
    arr = np.frombuffer(BIN, dtype=dt, count=a['count'] * n, offset=off).reshape(a['count'], n)
    if a.get('normalized'):
        arr = arr.astype(np.float32) / np.iinfo(dt).max
    return arr

def quat_m(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def local(n):
    M = np.eye(4)
    s = n.get('scale', [1, 1, 1])
    M[:3, :3] = quat_m(n.get('rotation', [0, 0, 0, 1])) @ np.diag(s)
    M[:3, 3] = n.get('translation', [0, 0, 0])
    return M

nodes = G['nodes']
parent = {}
for i, n in enumerate(nodes):
    for c in n.get('children', []):
        parent[c] = i
name = {n.get('name'): i for i, n in enumerate(nodes)}

def axis_angle(ax, a):
    ax = np.array(ax, float); ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K

# 姿勢の指定：ボーン名 → 「子ボーンをこの向きに向ける」（モデルの座標：Yが上、Zが前、Xが左）
AIM = {}
TWIST = {}
def compute(aims):
    glob = {}
    order = []
    def visit(i):
        order.append(i)
        for c in nodes[i].get('children', []):
            visit(c)
    root = [i for i in range(len(nodes)) if i not in parent and 'children' in nodes[i]]
    for r in root:
        visit(r)
    for i in order:
        P = glob[parent[i]] if i in parent else np.eye(4)
        M = P @ local(nodes[i])
        nm = nodes[i].get('name')
        if nm in aims:
            child, tgt = aims[nm]
            # 自分の位置と、子ボーンの今の位置から向きを出し、目標の向きへ回す
            c = name[child]
            Cm = M @ local(nodes[c])
            p = M[:3, 3]; d = Cm[:3, 3] - p; d /= np.linalg.norm(d)
            t = np.array(tgt, float); t /= np.linalg.norm(t)
            ax = np.cross(d, t); s = np.linalg.norm(ax); co = np.dot(d, t)
            if s > 1e-6:
                R = axis_angle(ax, np.arctan2(s, co))
                M2 = M.copy(); M2[:3, :3] = R @ M[:3, :3]
                M = M2
        glob[i] = M
    return glob

rest = compute({})
def pos(g, nm):
    return g[name[nm]][:3, 3]

for nm in ['pelvis', 'spine_03', 'Head', 'upperarm_l', 'lowerarm_l', 'hand_l', 'thigh_l', 'calf_l', 'foot_l']:
    print('rest', nm, np.round(pos(rest, nm), 3))

POSE = {
    # 太もも：ひざを少し前に出して床につく（正座に近い、艇の中で膝をつく姿勢）
    'thigh_l': ('calf_l', [0.06, -0.42, 0.9]),
    'thigh_r': ('calf_r', [-0.06, -0.42, 0.9]),
    # すね：後ろへ折りたたむ
    'calf_l': ('foot_l', [0.0, -0.3, -1.0]),
    'calf_r': ('foot_r', [0.0, -0.3, -1.0]),
    'foot_l': ('ball_l', [0.0, -1.0, -0.2]),
    'foot_r': ('ball_r', [0.0, -1.0, -0.2]),
    # 上体：前に倒す
    'spine_01': ('spine_02', [0.0, 0.85, 0.53]),
    'spine_02': ('spine_03', [0.0, 0.8, 0.6]),
    'spine_03': ('neck_01', [0.0, 0.78, 0.63]),
    'neck_01': ('Head', [0.0, 0.92, 0.38]),
    # 腕：前へ伸ばしてハンドルとレバーを握る
    'upperarm_l': ('lowerarm_l', [0.25, -0.35, 0.9]),
    'upperarm_r': ('lowerarm_r', [-0.25, -0.35, 0.9]),
    'lowerarm_l': ('hand_l', [-0.3, -0.15, 0.94]),
    'lowerarm_r': ('hand_r', [0.3, -0.15, 0.94]),
    'hand_l': ('middle_01_l', [-0.15, -0.3, 0.94]),
    'hand_r': ('middle_01_r', [0.15, -0.3, 0.94]),
}
# 指を軽く握る
for side in 'lr':
    for f in ['index', 'middle', 'ring', 'pinky']:
        POSE[f + '_02_' + side] = (f + '_03_' + side, [0, -1, 0.15])
posed = compute(POSE)
for nm in ['pelvis', 'Head', 'hand_l', 'hand_r', 'calf_l', 'foot_l']:
    print('posed', nm, np.round(pos(posed, nm), 3))

skin = G['skins'][0]
joints = skin['joints']
IBM = acc(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)  # 列優先 → 行優先
J = np.array([posed[j] @ IBM[k] for k, j in enumerate(joints)])

prim = G['meshes'][2]['primitives'][0]
P = acc(prim['attributes']['POSITION']).astype(np.float64)
N = acc(prim['attributes']['NORMAL']).astype(np.float64)
JI = acc(prim['attributes']['JOINTS_0']).astype(int)
W = acc(prim['attributes']['WEIGHTS_0']).astype(np.float64)
IDX = acc(prim['indices']).reshape(-1).astype(np.uint16)
W = W / W.sum(1, keepdims=True)

Ph = np.c_[P, np.ones(len(P))]
outP = np.zeros_like(P); outN = np.zeros_like(N)
for k in range(4):
    Mk = J[JI[:, k]]
    outP += W[:, k:k+1] * np.einsum('nij,nj->ni', Mk, Ph)[:, :3]
    outN += W[:, k:k+1] * np.einsum('nij,nj->ni', Mk[:, :3, :3], N)
outN /= np.linalg.norm(outN, axis=1, keepdims=True) + 1e-9

# 体の部位（いちばん影響の大きいボーン）で塗り分ける：0=勝負服 1=手袋 2=ブーツ 3=頭(ヘルメットで隠れる)
dom = JI[np.arange(len(JI)), W.argmax(1)]
jn = [nodes[joints[d]].get('name', '') for d in dom]
region = np.zeros(len(P), np.uint8)
for i, n in enumerate(jn):
    if any(f in n for f in ['hand', 'thumb', 'index', 'middle', 'ring', 'pinky']): region[i] = 1
    elif 'foot' in n or 'ball' in n: region[i] = 2
    elif n in ('Head', 'neck_01'): region[i] = 3

# ゲームの座標へ：前=+x、横=y、上=+z。単位は1m=8.6
S = 8.6
gx = outP[:, 2] * S; gy = outP[:, 0] * S; gz = outP[:, 1] * S
nx = outN[:, 2]; ny = outN[:, 0]; nz = outN[:, 1]
head = posed[name['Head']][:3, 3]
hx, hy, hz = head[2] * S, head[0] * S, head[1] * S
# 頭（ヘルメットの中心）が操縦席の上、ひざが艇の床に来るように位置を合わせる
knee = posed[name['calf_l']][:3, 3]
kz = knee[1] * S
pel = posed[name['pelvis']][:3, 3]
dx = -4.9 - pel[2] * S; dz = 1.9 - kz
gx += dx; gz += dz; hx += dx; hz += dz
print('head center', round(hx, 2), round(hy, 2), round(hz, 2), 'z range', round(gz.min(), 2), round(gz.max(), 2), 'x range', round(gx.min(), 2), round(gx.max(), 2))
helmet_c = (hx + 0.25, hy, hz + 0.9)

with open(sys.argv[1] if len(sys.argv) > 1 else 'rider.bin', 'wb') as f:
    nv, ni = len(P), len(IDX)
    f.write(struct.pack('<4sII4f', b'TKR1', nv, ni, helmet_c[0], helmet_c[1], helmet_c[2], 2.15))
    f.write(np.c_[gx, gy, gz].astype(np.float32).tobytes())
    q = np.clip(np.round(np.c_[nx, ny, nz] * 127), -127, 127).astype(np.int8)
    f.write(q.tobytes())
    f.write(region.tobytes())
    pad = (-(32 + nv * 12 + nv * 3 + nv)) % 2
    f.write(b'\0' * pad)
    f.write(IDX.tobytes())
np.save('/tmp/riderpts.npy', np.c_[gx, gy, gz, region])
print('verts', nv, 'tris', ni // 3)
