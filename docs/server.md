# 服务器连接与开发环境

验证日期：2026-11-20（北京时间）。

**当前可用入口为 `root@179.46.2.2:22`，已完成 SSH 登录和 CPU 算子环境验证。** 先接通 UniVPN，再执行 SSH。原表中另一台鲲鹏服务器 `179.125.7.2:22` 本次连接超时。

## 快速连接

1. 打开 UniVPN，选择 `HiDevLab`，连接 SSL VPN 网关 `123.60.114.227:41024`。
2. 使用本地环境信息表中「SSL VPN」的账号和密码登录，按客户端提示完成动态验证码验证。
3. 客户端显示登录成功后，在本机终端执行：

```bash
ssh -p 22 -o ConnectTimeout=10 root@179.46.2.2
```

SSH 密码使用环境信息表中 **业务系统 IP 为 `179.46.2.2` 的那一行**的「业务系统 密码」，与 VPN 密码分开。

4. 登录服务器后，加载已验证的开发环境：

```bash
source /opt/kunpeng-operator-lab/env.sh
cd /opt/kunpeng-operator-lab
```

## 连接参数与有效期

设备名称沿用原始分配表；开发环境状态来自本次实机检查。

| 项目 | 当前可用服务器 | 另一台分配服务器 |
| --- | --- | --- |
| 原表设备名称 | Atlas 800T A2 训练服务器 | 鲲鹏 920 |
| 业务系统 IP | `179.46.2.2` | `179.125.7.2` |
| SSH 用户 | `root`，已验证 | `root`，来自分配表 |
| SSH 端口与状态 | `22`，已登录 | `22`，VPN 接通后仍超时 |
| BMC 管理地址 | `https://190.90.46.11` | `https://190.90.125.15` |
| 分配开始时间 | 2026-11-16 19:56:46 | 2026-11-16 19:56:46 |
| 分配到期时间 | 2026-12-01 23:59:59 | 2026-12-01 23:59:59 |

到期或重新分配后，需从 HiDevLab 项目环境详情重新获取地址和凭据，并更新本文。

原始信息保存在本地 `HiDevLab环境信息_20261116000020.xlsx` 的「设备信息」工作表：

| 用途 | 读取位置 |
| --- | --- |
| VPN 网关、端口 | 「SSL VPN」组的 IP、端口，I/J 列 |
| VPN 账号、密码 | 「SSL VPN」组的用户名、密码，K/L 列 |
| SSH 地址、账号、密码 | 「业务系统」IP、用户名、密码，P/Q/R 列 |
| BMC 地址、账号、密码 | BMC IP、用户名、密码，M/N/O 列 |

密码、VPN 用户名和原始凭据表仅保存在本地；仓库的 `.gitignore` 已排除凭据表、原始附件、证书私钥和 Office 临时文件。

## UniVPN 设置

| 设置 | 值 |
| --- | --- |
| 客户端 | UniVPN；本机验证版本 `10781.19.0.1214` |
| 连接类型 | SSL VPN |
| 连接名称 | `HiDevLab` |
| 远程网关 | `123.60.114.227` |
| 端口 | `41024` |

首次配置时，通过「新建连接」选择 SSL VPN，填写上述网关与端口。密码含特殊符号时使用英文半角字符。动态验证码按提示从本人手机或邮箱获取。

本次网关出现不可信证书告警，经维护者确认后继续了本次连接，客户端证书告警设置保持开启。观测到的证书信息如下，供后续比对；证书信任应向环境管理员核实。

- Subject：`CN=LOCAL-101930058418`
- Issuer：`CN=Self-Sign-Cert-101930058418`
- SHA-256：`b0c465ef8a347dbf846cfd723db444e2b239d393d277740b1f3971e90a59938d`

操作参考：[华为 HiDevLab VPN 接入说明](https://www.hikunpeng.com/document/detail/zh/techcert/certtest/envapp/kunpeng_hidevlabeg_0012.html)、[UniVPN 官方下载页](https://www.leagsoft.com/doc/article/103107.html)。

## SSH 别名与文件传输

可将以下配置加入本机 `~/.ssh/config`：

```sshconfig
Host huawei-operator huawei-ascend
    HostName 179.46.2.2
    User root
    Port 22
    ConnectTimeout 10
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

配置后使用 `ssh huawei-operator`。该别名指向已验证的算子开发环境。

本次首次连接时观测到的 SSH ED25519 主机指纹：

```text
SHA256:EdKvDUjCKCcGa2gEtGKNsnCu2yFkZXNkbSJBPiD/k0Y
```

上传或下载文件同样需要保持 VPN 连接：

```bash
# 本机执行；remote_path 替换为实际远端文件路径
scp ./local_file root@179.46.2.2:/opt/kunpeng-operator-lab/src/
scp root@179.46.2.2:/remote_path ./
```

## 已验证的 CPU 算子开发环境

以下配置均位于 **`179.46.2.2`**，已通过实机登录检查。

| 项目 | 实测结果 |
| --- | --- |
| 操作系统 | Kylin Linux Advanced Server V10（Halberd） |
| 架构 | `aarch64` |
| CPU | Kunpeng 920，4 路、256 个逻辑 CPU |
| 内存 | 约 2.0 TiB |
| 开发目录 | `/opt/kunpeng-operator-lab` |
| 完整环境入口 | `/opt/kunpeng-operator-lab/env.sh` |
| Python 虚拟环境 | `/opt/kunpeng-operator-env` |
| Python / pip | `3.7.9` / `20.2.2` |
| NumPy | `1.21.6`；已通过 2×2 矩阵乘法检查 |
| GCC / CMake | `7.3.0` / `3.16.5` |

每次新建 SSH 会话，在服务器终端执行：

```bash
source /opt/kunpeng-operator-lab/env.sh
command -v python
python --version
python -m pip --version
printf 'VIRTUAL_ENV=%s\n' "$VIRTUAL_ENV"
```

完整入口会激活 Python 环境，并设置 OpenMP 线程数为 4、线程上限为 8，OpenBLAS/GotoBLAS 线程数为 1，临时目录为 `/opt/kunpeng-operator-lab/tmp`。

只需 Python 环境时，可执行 `source /opt/kunpeng-operator-env/bin/activate`；使用 `deactivate` 退出。仅激活 Python 环境时不包含上述线程和临时目录设置。

本次未检测到 SciPy、PyTorch、pytest。PCIe 列表中可见华为加速设备，但未发现 `npu-smi`、`/usr/local/Ascend` 或 `/dev/davinci0`；当前已验证的是 CPU 算子环境，NPU 驱动及 CANN 工具链仍需单独确认。

## 验证记录与排障

| 检查项 | 2026-11-20 结果 |
| --- | --- |
| VPN 网关 `123.60.114.227:41024` | TCP/TLS 可达；客户端登录成功 |
| 到业务网段的路由 | 接通后走 `utun6`；隧道接口编号随会话变化 |
| `179.46.2.2:22` | SSH 登录成功；使用原表对应行的业务系统凭据 |
| 环境脚本与 Python | 激活成功，路径和版本已核实，NumPy 计算通过 |
| `179.125.7.2:22` | 隧道接通后 TCP/SSH 仍超时 |
| 鲲鹏 BMC `190.90.125.15` | Redfish 只读查询成功；机型 `S920X20`，电源 `On`，健康状态 `Critical` |
| Atlas BMC `190.90.46.11` | HTTPS 可达；原表 BMC 凭据的 Redfish 请求返回 `401`，管理登录未验证 |

连接失败时，按阶段检查：

- **VPN 认证失败**：核对使用的是 SSL VPN 账号和密码，确认字符输入完整、账号与环境仍有效。
- **SSH 超时**：先确认 UniVPN 已连接。在 macOS 本机执行 `route -n get 179.46.2.2`，确认目标经过 VPN 隧道，再执行 `nc -vz -G 5 179.46.2.2 22`。
- **SSH 提示 Permission denied**：核对 IP 对应行的业务系统密码；若首次登录后已修改，应使用修改后的密码。
- **另一台鲲鹏服务器仍超时**：本次已确认其开机且 BMC 健康状态为 `Critical`，具体超时原因待查；后续可通过 BMC 控制台检查系统启动、业务网卡与 SSH 服务。
- **找不到开发环境**：先确认登录目标是 `179.46.2.2`，再检查 `/opt/kunpeng-operator-lab/env.sh`。
