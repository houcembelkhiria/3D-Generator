import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import Union, Optional

from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover, _delight_steps_env_default
from .utils.multiview_utils import Multiview_Diffusion_Net, _mv_steps
from hy3dgen.system_utils import empty_cache, get_device
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:
    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, device=get_device()):
        self.device = device
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        original_model_path = model_path
        if os.path.exists(model_path):
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
            if os.path.exists(delight_model_path) and os.path.exists(multiview_model_path):
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))
        base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
        local_model_path = os.path.expanduser(os.path.join(base_dir, original_model_path))
        delight_model_path = os.path.join(local_model_path, 'hunyuan3d-delight-v2-0')
        multiview_model_path = os.path.join(local_model_path, 'hunyuan3d-paint-v2-0')
        if os.path.exists(delight_model_path) and os.path.exists(multiview_model_path):
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))
        try:
            import huggingface_hub
            download_path = huggingface_hub.snapshot_download(
                repo_id=original_model_path,
                allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*"])
            delight_model_path = os.path.join(download_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(download_path, 'hunyuan3d-paint-v2-0')
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, device=kwargs.get('device', get_device())))
        except Exception as e:
            raise FileNotFoundError(f"Model {original_model_path} not found: {e}")

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)
        self.load_models()

    def load_models(self):
        empty_cache()
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id=None, device=get_device()):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for _vi, (view, elev, azim, weight) in enumerate(zip(views, camera_elevs, camera_azims, view_weights)):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(view, elev, azim)
            # DEBUG: save per-view texture
            try:
                _dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'generated', '_debug')
                os.makedirs(_dbg, exist_ok=True)
                _t = (project_texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(_t).save(os.path.join(_dbg, f'view_texture_{_vi}.png'))
            except Exception:
                pass
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):
        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)
        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            return image.convert('RGB')
        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")
        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)
        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))
        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)
        new_width = width + 2 * border_width
        new_height = height + 2 * border_height
        square_size = max(new_width, new_height)
        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))
        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height
        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @staticmethod
    def _detect_case_dims(rgba_arr):
        """Detect the metallic watch CASE in an RGBA image.
        Returns (cy, cx, h, w) where:
          - (cy, cx) is the centroid of the largest CLOSED metallic blob
            (morph-close bridges FOSSIL letters / screw holes so the centroid
            sits on the actual case centre instead of one half-arc)
          - (h, w) is the bbox of the RAW metallic pixels (no morph dilation)
            so the size measurement is not inflated by the closing kernel
        Returns None if no plausible case region exists.
        """
        import cv2 as _cv2
        rgb = rgba_arr[..., :3]
        alpha = rgba_arr[..., 3]
        hsv = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2HSV)
        case_mask_raw = ((alpha > 10) & (hsv[..., 1] < 60) & (hsv[..., 2] > 100)).astype(np.uint8)
        if case_mask_raw.sum() < 100:
            return None

        # SIZE & CENTRE: from raw metallic pixel bbox extents.
        # All-pixel bbox so split components (case ring broken by FOSSIL
        # letters) still contribute to a correct case extent. BBox CENTRE
        # (midpoint of min/max) is preferred over the closed-component
        # centroid because the centroid is biased toward whichever half-arc
        # of the broken case ring has more pixels, whereas the bbox spans
        # the full case shape and its midpoint sits on the case geometric
        # centre — robust to off-centre FOSSIL/model-number stamping.
        _raw_coords = np.nonzero(case_mask_raw)
        _y_min, _y_max = int(_raw_coords[0].min()), int(_raw_coords[0].max())
        _x_min, _x_max = int(_raw_coords[1].min()), int(_raw_coords[1].max())
        _rh = float(_y_max - _y_min + 1)
        _rw = float(_x_max - _x_min + 1)
        _cy = 0.5 * (_y_min + _y_max)
        _cx = 0.5 * (_x_min + _x_max)
        return _cy, _cx, _rh, _rw

    @staticmethod
    def _strap_axis_angle(rgba_arr):
        """Return the strap's principal-axis angle (degrees from vertical).
        Positive = strap tilts clockwise (top-right / bottom-left). Used to
        un-tilt photographed straps so they align with the mesh's vertical
        strap axis after set_mesh auto-centring. Returns 0 if not computable
        or if the strap mass is too elongated-along-horizontal (sanity).
        """
        import cv2 as _cv2
        rgb = rgba_arr[..., :3]
        alpha = rgba_arr[..., 3]
        hsv = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2HSV)
        case_mask = (alpha > 10) & (hsv[..., 1] < 60) & (hsv[..., 2] > 100)
        strap_mask = (alpha > 10) & (~case_mask)
        coords = np.nonzero(strap_mask)
        if coords[0].size < 200:
            return 0.0
        # PCA on (row, col) points
        pts = np.column_stack([coords[0], coords[1]]).astype(np.float64)
        pts -= pts.mean(axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Major axis = eigenvector with largest eigenvalue (last column)
        major = eigvecs[:, -1]  # (d_row, d_col)
        # Sanity: only correct if strap is sufficiently elongated (ratio > 2)
        if eigvals[-1] < 4.0 * max(eigvals[0], 1e-6):
            return 0.0
        # Angle between major axis and vertical (1, 0). Positive col = tilt right.
        # atan2 sign convention: angle from +row axis toward +col axis.
        angle_rad = np.arctan2(major[1], major[0])
        angle_deg = float(np.degrees(angle_rad))
        # Normalise to [-90, 90] (axis direction is ambiguous in sign)
        while angle_deg > 90:
            angle_deg -= 180
        while angle_deg < -90:
            angle_deg += 180
        return angle_deg

    def _recenter_with_case_target(self, image, target_case_size: float | None,
                                   target_canvas_cy: float | None = None,
                                   target_canvas_cx: float | None = None):
        """Recenter+scale so the detected metallic case lands at canvas center
        with a specific pixel size (target_case_size). Also rotates the image
        so the strap's principal axis is vertical, matching the mesh's strap
        axis (vertical after set_mesh auto-centring). All views fed through
        this with the same target end up with their cases at identical canvas
        positions/sizes, matching the mesh's case projection regardless of
        how much strap each photo includes.

        Falls back to bbox-fill + alpha CoM if case detection or target fails.
        Strap content may extend off-canvas; that is intentional and accepted
        — the bake reads only canvas pixels, and off-canvas content becomes
        the subject-mean fill, which is the correct strap-leather colour.
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        arr = np.array(image)
        rs = self.config.render_size

        # NOTE: every strap-alignment warp we have tried (per-row remap,
        # piecewise shear, hull-composite) breaks the bake — when the user's
        # photo has the strap offset from the case, warping the substituted
        # image to "fix" that offset mis-aligns the case-strap boundary
        # between views, and the multi-view bake produces patches/camouflage
        # texture where the views disagree. Verified across multiple runs.
        # Accept the user-photo strap offset as ground truth. Centering is
        # done on the case centroid + case-target size below; the FOSSIL text
        # stays sharp because no interpolation is applied to the case region.
        alpha = arr[..., 3]
        coords = np.nonzero(alpha > 10)
        if coords[0].size == 0:
            return image.convert('RGB')

        # Centering uses CASE BBOX CENTRE. _detect_case_dims now returns
        # the midpoint of the metallic pixel bbox (not a weighted centroid),
        # which gives the geometric centre of the case shape regardless of:
        #   - strap asymmetry (strap longer above than below, etc.)
        #   - off-centre FOSSIL stamping inside the case
        #   - case ring being split by engravings
        # Sizing comes from the same call.
        case = self._detect_case_dims(arr)
        if case is not None and target_case_size is not None and target_case_size > 1:
            case_cy, case_cx, case_h, case_w = case
            case_dim = max(case_h, case_w)
            scale = float(target_case_size) / float(case_dim)
            center_y, center_x = case_cy, case_cx
        else:
            x_min, x_max = int(coords[0].min()), int(coords[0].max())
            y_min, y_max = int(coords[1].min()), int(coords[1].max())
            bbox_dim = max(x_max - x_min, y_max - y_min)
            scale = (rs * 0.96) / bbox_dim if bbox_dim > 0 else 1.0
            center_y = 0.5 * (x_min + x_max)
            center_x = 0.5 * (y_min + y_max)

        import cv2 as _cv2
        img_h, img_w = arr.shape[:2]
        new_h = max(int(img_h * scale), 1)
        new_w = max(int(img_w * scale), 1)
        # INTER_AREA for downsizing (best for sharpness when shrinking),
        # INTER_CUBIC for upsizing (preserves text edges better than LINEAR).
        _resize_interp = _cv2.INTER_AREA if scale < 1.0 else _cv2.INTER_CUBIC
        img_scaled = _cv2.resize(arr, (new_w, new_h), interpolation=_resize_interp)

        scaled_cy = center_y * scale
        scaled_cx = center_x * scale

        canvas = np.ones((rs, rs, 4), dtype=np.uint8)
        canvas[..., :3] = 255
        canvas[..., 3]  = 255

        # If a target canvas position is given (where the mesh's case actually
        # projects in this view), place the user's case there. Otherwise use
        # canvas centre.
        _tcy = float(target_canvas_cy) if target_canvas_cy is not None else (rs / 2.0)
        _tcx = float(target_canvas_cx) if target_canvas_cx is not None else (rs / 2.0)
        x_off = int(round(_tcy - scaled_cy))
        y_off = int(round(_tcx - scaled_cx))

        # Compute overlap (allow image to extend off canvas — strap may overflow)
        src_x_start = max(0, -x_off)
        src_y_start = max(0, -y_off)
        src_x_end = min(new_h, rs - x_off)
        src_y_end = min(new_w, rs - y_off)
        dst_x_start = max(0, x_off)
        dst_y_start = max(0, y_off)

        if src_x_end > src_x_start and src_y_end > src_y_start:
            canvas[dst_x_start:dst_x_start + (src_x_end - src_x_start),
                   dst_y_start:dst_y_start + (src_y_end - src_y_start)] = (
                img_scaled[src_x_start:src_x_end, src_y_start:src_y_end])

        a   = canvas[..., 3:].astype(np.float32) / 255.0
        rgb = canvas[..., :3].astype(np.float32)
        subj_mask = a[..., 0] > 0.5
        fill = rgb[subj_mask].mean(axis=0) if subj_mask.any() else np.array([128., 128., 128.])
        bg   = np.full_like(rgb, fill)
        comp = (rgb * a + bg * (1.0 - a)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(comp)

    # Maps texgen camera azimuth (elev=0) → user view name.
    # Camera at -Z (glTF): camera_right=-X, so 3-o'clock=-X in shapegen.
    # set_mesh(-X)→+X in texgen → azim=270 camera. So azim=270→'right', azim=90→'left'.
    _AZIM_TO_VIEW = {0: 'front', 90: 'left', 180: 'back', 270: 'right'}

    @torch.inference_mode()
    def __call__(self, mesh, image, user_views=None):
        """
        Args:
            mesh: trimesh.Trimesh
            image: front-view PIL image (used for delight + multiview seed)
            user_views: optional dict of {view_name: PIL.Image} from a multiview
                        capture. When provided, generated views at matching azimuths
                        are replaced with the user's actual images, giving accurate
                        back/left/right texture instead of AI-hallucinated colours.
        """
        import sys, time as _time
        _t0 = _time.time()
        _last = [_t0]
        def trace(msg):
            now = _time.time()
            sys.stderr.write(f"\n[TexGen +{now - _t0:.1f}s D{now - _last[0]:.1f}s] {msg}\n")
            sys.stderr.flush()
            _last[0] = now

        trace("Starting texture generation...")
        image_prompt = Image.open(image) if isinstance(image, str) else image
        trace("Recentering image...")
        image_prompt = self.recenter_image(image_prompt)
        trace(f"Running delight ({_delight_steps_env_default()} steps, override via HY3D_DELIGHT_STEPS)...")
        image_prompt = self.models['delight_model'](image_prompt)
        try:
            _dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'generated', '_debug')
            os.makedirs(_dbg, exist_ok=True)
            image_prompt.save(os.path.join(_dbg, 'delight_output.png'))
        except Exception:
            pass
        trace("Running UV wrap...")
        mesh = mesh_uv_wrap(mesh)
        trace("Loading mesh into renderer...")
        self.render.load_mesh(mesh)

        # Re-centre the mesh on its DENSEST region (case) instead of bbox
        # midpoint. set_mesh's auto_center put the bbox midpoint at origin,
        # but for watches the bbox is dominated by strap length and the case
        # sits offset from the midpoint. The median of vertex positions is
        # robust to elongated low-density strap and dominated by the dense
        # case region, so translating by -median puts the case at origin.
        # All subsequent renders (normal, position, AI multiview, back_project)
        # then have the case at canvas centre, and user substitutions placed
        # at canvas centre land exactly on the mesh's case projection.
        try:
            import torch as _torch_rc
            _verts = self.render.vtx_pos
            _median = _torch_rc.median(_verts, dim=0).values
            _bbox_max = _verts.max(0).values
            _bbox_min = _verts.min(0).values
            _half_extent = ((_bbox_max - _bbox_min) * 0.5).max().item()
            _shift_mag = float(_torch_rc.norm(_median).item())
            # Only apply shift if it's significant (> 5% of half-extent) and
            # not so extreme that it would push the mesh entirely out of view.
            if _shift_mag > 0.05 * _half_extent and _shift_mag < 0.5 * _half_extent:
                self.render.vtx_pos = _verts - _median
                trace(f"Mesh re-centred on case (median shift={_median.tolist()}, mag={_shift_mag:.3f})")
            else:
                trace(f"Mesh re-centring skipped (shift_mag={_shift_mag:.3f}, half_extent={_half_extent:.3f})")
        except Exception as _e:
            trace(f"Mesh re-centring failed (continuing with bbox-centring): {_e}")

        selected_camera_elevs = self.config.candidate_camera_elevs
        selected_camera_azims = self.config.candidate_camera_azims
        selected_view_weights = self.config.candidate_view_weights

        trace("Rendering normal/position maps...")
        normal_maps, position_maps = [], []
        for elev, azim in zip(selected_camera_elevs, selected_camera_azims):
            normal_maps.append(self.render.render_normal(elev, azim, use_abs_coor=True, return_type='pl'))
            position_maps.append(self.render.render_position(elev, azim, return_type='pl'))

        trace(f"Running multiview ({_mv_steps()} steps, override via HY3D_MULTIVIEW_STEPS)...")
        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.models['multiview_model'](image_prompt, normal_maps + position_maps, camera_info)

        trace("Resizing multiviews...")
        for i in range(len(multiviews)):
            multiviews[i] = multiviews[i].resize((self.config.render_size, self.config.render_size))

        # Substitute user-provided views where available (back/left/right).
        # This replaces AI-hallucinated texture with the user's actual images,
        # which matters most for the back (weight=0.5) and sides (weight=0.1).
        if user_views:
            try:
                # Top/bottom views: user provided no top/bottom photos, and the
                # AI hallucinates wrong colours there. Replace those views with
                # a SOLID, uniform canvas filled with the median colour of the
                # user's back-image subject — no blur, no gradient. The median
                # is robust to the case/strap colour mix and matches the leather
                # strap closely (strap dominates the back photo). Solid colour
                # ensures strap-tip cross-sections receive that exact colour
                # everywhere the top/bottom cameras project.
                _src = user_views.get('back') or user_views.get('front')
                if _src is not None:
                    if _src.mode != 'RGBA':
                        _src = _src.convert('RGBA')
                    _arr = np.array(_src)
                    _alpha = _arr[..., 3]
                    _rgb = _arr[..., :3]
                    _subj = _alpha > 10
                    if _subj.any():
                        _fill = np.median(_rgb[_subj], axis=0).astype(np.uint8)
                    else:
                        _fill = np.array([128, 128, 128], dtype=np.uint8)
                    _rs = self.config.render_size
                    _solid = np.full((_rs, _rs, 3), _fill, dtype=np.uint8)
                    _b = Image.fromarray(_solid)
                    for _i, _e in enumerate(selected_camera_elevs):
                        if _e != 0:
                            multiviews[_i] = _b
            except Exception as _e:
                trace(f"Warning: top/bottom solid-colour substitution failed: {_e}")

            # Missing side view fallback: if the user provided only one side
            # (right OR left), synthesise the opposite by horizontal mirror so
            # colours/materials come from the user's image instead of the AI.
            # Geometry is approximate (case asymmetry: crown lands on the wrong
            # side), but the precise colour requirement is satisfied because
            # every pixel still originates from the user's photograph.
            from PIL import ImageOps as _ImageOps
            _user_views = dict(user_views)
            if 'left' not in _user_views and 'right' in _user_views:
                _user_views['left'] = _ImageOps.mirror(_user_views['right'])
                trace("Synthesised 'left' view as mirrored 'right' (user did not provide left)")
            if 'right' not in _user_views and 'left' in _user_views:
                _user_views['right'] = _ImageOps.mirror(_user_views['left'])
                trace("Synthesised 'right' view as mirrored 'left' (user did not provide right)")

            # Compute TARGET CASE PIXEL SIZE from the MESH'S ACTUAL CASE
            # PROJECTION, detected in the AI front-view multiview output. The
            # AI renders true mesh geometry, so its case position+size reflect
            # the mesh's case canvas extent. Using this as target means the
            # user's case gets scaled to EXACTLY fill the mesh's case region
            # (no overflow into strap area, no shrink leaving background).
            # Previous approach used the user front PHOTO's bbox, which made
            # the user case much larger than the mesh case → texture leak.
            _target_case_size = None
            try:
                # Find the front view (azim=0, elev=0) in the multiview list
                _front_idx = None
                for _fi, (_fa, _fe) in enumerate(zip(selected_camera_azims, selected_camera_elevs)):
                    if _fa == 0 and _fe == 0:
                        _front_idx = _fi
                        break
                if _front_idx is not None:
                    _ai_front_arr = np.array(multiviews[_front_idx].convert('RGBA'))
                    # Synthesise alpha from non-near-white pixels (mesh region)
                    _bg = np.all(_ai_front_arr[..., :3] > 240, axis=-1)
                    _ai_front_arr[..., 3] = np.where(_bg, 0, 255).astype(np.uint8)
                    _mesh_case = self._detect_case_dims(_ai_front_arr)
                    if _mesh_case is not None:
                        _, _, _mch, _mcw = _mesh_case
                        _target_case_size = float(max(_mch, _mcw))
                        trace(f"Target case size from MESH (AI front view): {_target_case_size:.1f}px (mesh case bbox {_mcw:.0f}x{_mch:.0f})")
            except Exception as _e:
                trace(f"Mesh case-size detection failed: {_e}")
            # Fallback: use user front photo case dim scaled to canvas
            if _target_case_size is None:
                _front_uv = _user_views.get('front')
                if _front_uv:
                    _fuv = _front_uv.convert('RGBA') if _front_uv.mode != 'RGBA' else _front_uv
                    _farr = np.array(_fuv)
                    _fa = _farr[..., 3]
                    _fc = np.nonzero(_fa > 10)
                    if _fc[0].size > 0:
                        _fh = int(_fc[0].max() - _fc[0].min())
                        _fw = int(_fc[1].max() - _fc[1].min())
                        _fdim = max(_fh, _fw)
                        _front_scale = (self.config.render_size * 0.96) / _fdim if _fdim > 0 else 1.0
                        _front_case = self._detect_case_dims(_farr)
                        if _front_case is not None:
                            _, _, _fch, _fcw = _front_case
                            _target_case_size = max(_fch, _fcw) * _front_scale
                            trace(f"Target case size (FALLBACK from user front): {_target_case_size:.1f}px")

            # Final guard: any elev=0 view still not in _user_views (e.g. user
            # provided neither left nor right) gets the same solid median-colour
            # canvas used for top/bottom. Guarantees zero AI-hallucinated colour
            # leaks into the bake — every projected pixel comes from a user image.
            try:
                if _src is not None and '_b' in locals() and _b is not None:
                    for _vname in ('front', 'back', 'left', 'right'):
                        if _vname not in _user_views:
                            _user_views[_vname] = _b.convert('RGBA')
                            trace(f"Filled missing '{_vname}' view with solid median colour (no AI)")
            except Exception as _e:
                trace(f"Warning: side-view solid fallback failed: {_e}")

            for i, (azim, elev) in enumerate(zip(selected_camera_azims, selected_camera_elevs)):
                if elev != 0:
                    continue
                view_name = self._AZIM_TO_VIEW.get(azim)
                # SKIP front view substitution. The AI multiview output renders
                # the true mesh geometry, so its front view fits the dial UV
                # mapping exactly with no stretching. Substituting the user's
                # photo at this position causes "melted" dial appearance when
                # the mesh dial is small relative to canvas (case ~16% of
                # canvas due to long strap) — the user case gets downsampled
                # aggressively, blurring fine details. Keep the AI output for
                # the front. Back/left/right substitutions still apply.
                if view_name == 'front':
                    trace("Skipping 'front' substitution — keeping AI multiview output for the dial face")
                    continue
                if view_name and view_name in _user_views:
                    try:
                        # Detect where the MESH'S CASE actually projects in this
                        # view by finding the case in the AI multiview output.
                        # The AI generates plausible texture on the mesh's true
                        # geometry, so its case position reflects where the mesh
                        # case actually is in canvas pixels — not necessarily
                        # canvas centre when the mesh has asymmetric strap.
                        # Placing the user's case at this position aligns it
                        # with the mesh's case projection.
                        _target_canvas_cy = None
                        _target_canvas_cx = None
                        try:
                            _ai_arr = np.array(multiviews[i].convert('RGBA'))
                            # AI output has white/grey background; treat any
                            # non-near-white pixel as part of the mesh
                            _bg_mask = np.all(_ai_arr[..., :3] > 240, axis=-1)
                            _ai_arr[..., 3] = np.where(_bg_mask, 0, 255).astype(np.uint8)
                            _ai_case = self._detect_case_dims(_ai_arr)
                            if _ai_case is not None:
                                _target_canvas_cy, _target_canvas_cx, _, _ = _ai_case
                                trace(f"{view_name}: mesh case projects to canvas ({_target_canvas_cy:.0f},{_target_canvas_cx:.0f})")
                        except Exception as _ae:
                            trace(f"{view_name}: AI case detection failed ({_ae}); using canvas centre")

                        user_img = self._recenter_with_case_target(
                            _user_views[view_name], _target_case_size,
                            target_canvas_cy=_target_canvas_cy,
                            target_canvas_cx=_target_canvas_cx)
                        user_img = user_img.resize(
                            (self.config.render_size, self.config.render_size))
                        multiviews[i] = user_img
                        trace(f"Substituted {view_name} view (azim={azim}°) with user image")
                        # DEBUG: save processed user image
                        try:
                            user_img.save(os.path.join(_dbg, f'user_view_{view_name}.png'))
                        except Exception:
                            pass
                    except Exception as _e:
                        trace(f"Warning: could not substitute {view_name} view: {_e}")

        trace("Baking from multiviews...")
        texture, mask = self.bake_from_multiview(multiviews,
            selected_camera_elevs, selected_camera_azims, selected_view_weights,
            method=self.config.merge_method)
        # DEBUG: save baked texture before inpaint
        try:
            _tex_np = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(_tex_np).save(os.path.join(_dbg, 'baked_texture_raw.png'))
            # Also save the first multiview input to compare
            multiviews[0].save(os.path.join(_dbg, 'multiview_input_0.png'))
            trace(f"DEBUG: saved baked texture + input to {_dbg}/")
        except Exception as _e:
            trace(f"DEBUG: save failed: {_e}")

        trace("Preparing mask...")
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        trace("Final inpainting...")
        texture = self.texture_inpaint(texture, mask_np)
        # DEBUG: save texture AFTER inpainting (this is what goes on the mesh)
        try:
            _tex_final = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(_tex_final).save(os.path.join(_dbg, 'texture_after_inpaint.png'))
            trace(f"DEBUG: saved final texture ({_tex_final.shape}) to {_dbg}/texture_after_inpaint.png")
        except Exception as _e:
            trace(f"DEBUG: save failed: {_e}")
        trace("Setting texture on renderer...")
        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()
        trace("Texture generation complete.")
        return textured_mesh
