import argparse
import os
import pymeshlab
import time


def fix_mesh(infile, outfile, alpha_fraction=0.001, stepsmoothnum=1, targetperc=0.6):
    """
    Fixes a mesh by removing isolated pieces, smoothing, applying alpha wrapping,
    and simplifying the mesh.
    
    Parameters:
    infile (str): Path to the input mesh file.
    outfile (str): Path to save the fixed mesh file. (default: 'mesh_fixed.ply')
    alpha_fraction (float): The size of the ball (fraction) for alpha wrapping. 
    stepsmoothnum (int): The number of times that the HC Laplacian smoothing algorithm is iterated.
    targetperc (float, 0...1): Target percentage reduction for mesh simplification.
    """

    # Create a MeshSet object
    ms = pymeshlab.MeshSet()

    # load the input mesh
    start_time = time.time()
    ms.load_new_mesh(infile)
    print(f"Input mesh loaded. Face count: {ms.current_mesh().face_number()}")

    # remove isolated pieces
    start_time = time.time()
    ms.meshing_remove_connected_component_by_diameter()
    print(f"✅ Removing isolated pieces. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # smooth the output mesh
    start_time = time.time()
    ms.apply_coord_two_steps_smoothing(normalthr=20.0, stepnormalnum=6, stepfitnum=6)
    print(f"✅ Smoothing mesh. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # create a new mesh object using alpha wrapping
    start_time = time.time()
    ms.generate_alpha_wrap(alpha_fraction=alpha_fraction, offset_fraction=0.000200)
    print(f"✅ Alpha wrapping. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # set the second mesh (alpha wrapping mesh) as current mesh
    ms.set_current_mesh(1)

    # HC Laplacian smoothing
    start_time = time.time()
    for _ in range(stepsmoothnum):
        ms.apply_coord_hc_laplacian_smoothing()
    print(f"✅ HC Laplacian smoothing. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # simplify the mesh
    start_time = time.time()
    ms.meshing_decimation_quadric_edge_collapse(targetperc=targetperc, preservetopology=False, planarquadric=True)
    print(f"✅ Mesh simplification. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # save the fixed mesh to outfile
    start_time = time.time()
    ms.save_current_mesh(outfile)
    print(f"Mesh fixed and saved to {outfile}. Face count: {ms.current_mesh().face_number()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh fixing utility using pymeshlab.")
    parser.add_argument('--infile', type=str, required=True, help='Input mesh file (required)')
    parser.add_argument('--outfile', type=str, default=None, help='Output mesh file (default: infile + _fixed)')
    parser.add_argument('--alpha_fraction', type=float, default=0.001, help='Alpha wrapping ball size fraction (default: 0.001)')
    parser.add_argument('--stepsmoothnum', type=int, default=1, help='HC Laplacian smoothing steps (default: 1)')
    parser.add_argument('--targetperc', type=float, default=0.6, help='Target percentage for mesh simplification (default: 0.6)')
    args = parser.parse_args()

    infile = args.infile
    if args.outfile is not None:
        outfile = args.outfile
    else:
        base, ext = os.path.splitext(infile)
        outfile = f"{base}_fixed{ext}"

    start_time = time.time()
    fix_mesh(
        infile,
        outfile,
        alpha_fraction=args.alpha_fraction,
        stepsmoothnum=args.stepsmoothnum,
        targetperc=args.targetperc
    )
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")
