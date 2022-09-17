
import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation


let metallib : String = (ProcessInfo.processInfo.environment["__BabelMetal"] ?? "the lat in the dictionary was nil!") + "/Babel.metallib"

// Defining global variables
var device:MTLDevice!
var commandQueue:MTLCommandQueue!
var computePipelineState_SnapShot:MTLComputePipelineState!
var computePipelineState_Sensors:MTLComputePipelineState!
var defaultLibrary:MTLLibrary!
let func_names:[String] = ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]
var particle_funcs:[MTLComputePipelineState?]! = []
var stress_funcs:[MTLComputePipelineState?]! = []
var ALL_arguments:MTLArgumentEncoder!
var Arguments_Buffer:MTLBuffer?

var mex_array: [Int: Int] = [:]

var mex_buffer:[MTLBuffer?] = []
// var uint_buffer:MTLBuffer?
// var index_mex:MTLBuffer?
// var index_uint:MTLBuffer?

var floatCounter:Int = 0

var stress_commandBuffer:MTLCommandBuffer!
var SnapShotsBuffer:MTLBuffer?

@_cdecl("InitializeMetalDevices")
public func InitializeMetalDevices() -> Int {   
    // Empties arrays from previous runs
    particle_funcs = []
    stress_funcs = []
    mex_array = [:]

    let devices = MTLCopyAllDevices()
    print("Found", devices.count, "METAL Devices!")
    if devices.count == 0 {
        print("No devices found! (How has this happened?)")
        return -1
    }

    print("Devices:")
    for d in devices {
        print(d.name)
        print("Is device low power? \(d.isLowPower).")
        print("Is device external? \(d.isRemovable).")
        print("Maximum threads per group: \(d.maxThreadsPerThreadgroup).")
        print("Maximum buffer length: \(Float(d.maxBufferLength) / 1024 / 1024 / 1024) GB.")
        print("")
    }

    print("The Device METAL selects as default is:", MTLCreateSystemDefaultDevice()!.name)
    let request : String = ProcessInfo.processInfo.environment["__BabelMetalDevice"]!
    print("Requested device ")
    print(request)
    if request != "" {
        var bFound = false
        for dev in MTLCopyAllDevices() {
            print(dev.name)
            if dev.name.contains(request)
                {
                print("Specified Device Found! Selecting device...")
                bFound = true
                device = dev
                break
                }
            }
            if bFound == false
            {
                print("Specified device NOT Found!")
                return -1
            }
    }
    else
    {
        print("No device specified, defaulting to system default device.")
        device = MTLCreateSystemDefaultDevice()!
    }

    commandQueue = device.makeCommandQueue()!
    defaultLibrary = try! device.makeLibrary(filepath: metallib)
  
    for x in func_names
    {
        var y = x
        y +=  "_ParticleKernel"
        let particle_function = defaultLibrary.makeFunction(name: y)!
        let particle_pipeline = try! device.makeComputePipelineState(function:particle_function) //Adjust try
        particle_funcs.append(particle_pipeline)
        y = x
        y +=  "_StressKernel"
        let stress_function = defaultLibrary.makeFunction(name: y)!
        let stress_pipeline = try! device.makeComputePipelineState(function:stress_function) //Adjust try
        stress_funcs.append(stress_pipeline)
        
    }

    print("Making Compute Pipeline State Objects for SnapShot and Sensors...")

    let SnapShotFunc = defaultLibrary.makeFunction(name: "SnapShot")!
    let SensorsKernelFunc = defaultLibrary.makeFunction(name: "SensorsKernel")!


    computePipelineState_SnapShot = try! device.makeComputePipelineState(function:SnapShotFunc)
    computePipelineState_Sensors = try! device.makeComputePipelineState(function:SensorsKernelFunc)

    ALL_arguments = SensorsKernelFunc.makeArgumentEncoder(bufferIndex: 0)
    Arguments_Buffer = device.makeBuffer(length: ALL_arguments.encodedLength, options: MTLResourceOptions())
    ALL_arguments?.setArgumentBuffer(Arguments_Buffer, offset: 0)  

    print("ALL_arguments.encodedLength",ALL_arguments.encodedLength)

    print("Function creation success!")
    return 0
}

@_cdecl("SymbolInitiation_uint")
public func SymbolInitiation_uint(index: UInt32, data: UInt32) -> Int {
    ALL_arguments?.constantData(at: Int(index)).storeBytes(of: data, as: UInt32.self)
    return 0
}

@_cdecl("SymbolInitiation_mex")
public func SymbolInitiation_mex(index: UInt32, data:Float32) -> Int{
    ALL_arguments?.constantData(at: Int(index)).storeBytes(of: data, as: Float32.self)
    return 0
}

@_cdecl("ownGpuCalloc") public func ownGpuCalloc(index: UInt32, size:Int) -> Int {
    
    let buffer = device.makeBuffer(length: size, options:MTLResourceOptions.storageModeManaged)
    mex_buffer.append(buffer)
    mex_array.updateValue(mex_buffer.count-1, forKey: Int(index))
    ALL_arguments?.setBuffer(buffer,  offset:0,index: Int(index))
    
    return 0
}

@_cdecl("CreateAndCopyFromMXVarOnGPUUINT") public func CreateAndCopyFromMXVarOnGPUUINT(index: UInt32, size:Int, ptr:UnsafeMutablePointer<UInt32> ) -> Int {
    let ll = MemoryLayout<UInt32>.stride * size
    let buffer = device.makeBuffer(bytes: ptr, length: ll, options:MTLResourceOptions.storageModeManaged)
    mex_buffer.append(buffer)
    mex_array.updateValue(mex_buffer.count-1, forKey: Int(index))
    ALL_arguments?.setBuffer(buffer, offset:0, index: Int(index))
    return 0
}

@_cdecl("CreateAndCopyFromMXVarOnGPUMex") public func CreateAndCopyFromMXVarOnGPUMex(index: UInt32, size:Int, ptr:UnsafeMutablePointer<Float32> ) -> Int {
    let ll = MemoryLayout<Float32>.stride * size
    let buffer = device.makeBuffer(bytes: ptr, length: ll, options:MTLResourceOptions.storageModeManaged)
    mex_buffer.append(buffer)
    mex_array.updateValue(mex_buffer.count-1, forKey: Int(index))
    ALL_arguments?.setBuffer(buffer,  offset:0,index: Int(index))
    return 0
}

@_cdecl("GetMaxTotalThreadsPerThreadgroup")
public func GetMaxTotalThreadsPerThreadgroup(fun:UnsafeRawPointer, id:Int) -> UInt32{
    let func_name = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)
    var index:Int!
    for name in func_names{
        if name.contains(func_name as! String){
        index = func_names.firstIndex(of: name)!
        break
        }
    }
    if id == 0{
        return UInt32(stress_funcs[index]!.maxTotalThreadsPerThreadgroup)
    }
    else{
        return UInt32(particle_funcs[index]!.maxTotalThreadsPerThreadgroup)
    }
}

@_cdecl("GetThreadExecutionWidth") 
public func GetThreadExecutionWidth(fun:UnsafeMutablePointer<CChar>, id:Int)-> UInt32 {
    let func_name = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)    
    var index:Int!
    for name in func_names{
        if name.contains(func_name as! String){
            index = func_names.firstIndex(of: name)!
            break
        }
    }

    if id == 0{
        return UInt32(stress_funcs[index]!.threadExecutionWidth)
    }
    else{
        return UInt32(particle_funcs[index]!.threadExecutionWidth)
    }
}

@_cdecl("SyncInput")
public func SyncInput() -> Int {
    let r:Range = 0..<ALL_arguments.encodedLength
    Arguments_Buffer!.didModifyRange(r)
    return 0
}

@_cdecl("EncoderInit")
public func EncoderInit(){
    stress_commandBuffer = commandQueue.makeCommandBuffer()!  
}

@_cdecl("EncodeStress")
public func EncodeStress(fun:UnsafeRawPointer, i:UInt32, j:UInt32, 
                        k:UInt32, x:UInt32, y:UInt32, z:UInt32,
                        Gx:UInt32,Gy:UInt32,Gz:UInt32){

    let func_name = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)
    var ind:Int!
    for name in func_names{
        if name.contains(func_name as! String){
        ind = func_names.firstIndex(of: name)!
        break
        }
    } 
    let computeCommandEncoder = stress_commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(Arguments_Buffer, offset: 0, index: 0)
    computeCommandEncoder.setComputePipelineState(stress_funcs[ind]!)
    //computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.dispatchThreads(MTLSize(width: Int(Gx), height: Int(Gy), depth: Int(Gz)),
                            threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.endEncoding()
}

@_cdecl("EncodeParticle")
public func EncodeParticle(fun:UnsafeRawPointer, i:UInt32, j:UInt32, k:UInt32, x:UInt32, y:UInt32, z:UInt32,
                            Gx:UInt32,Gy:UInt32,Gz:UInt32){
    let func_name = NSString(bytes:fun, length: 5, encoding:String.Encoding.utf8.rawValue)
    var index:Int!
    for name in func_names{
        if name.contains(func_name as! String){
        index = func_names.firstIndex(of: name)!
        break
        }
    }
    let computeCommandEncoder = stress_commandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(Arguments_Buffer, offset: 0, index: 0)
    computeCommandEncoder.setComputePipelineState(particle_funcs[index]!)
    //computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.dispatchThreads(MTLSize(width: Int(Gx), height: Int(Gy), depth: Int(Gz)),
                            threadsPerThreadgroup:MTLSize(width:Int(x), height: Int(y), depth: Int(z)))
    computeCommandEncoder.endEncoding()
}

@_cdecl("EncodeCommit")
public func EncodeCommit(){
    stress_commandBuffer.commit()
    stress_commandBuffer.waitUntilCompleted()
}


@_cdecl("CreateAndCopyFromMXVarOnGPUSnapShot")
public func CreateAndCopyFromMXVarOnGPUSnapShot(numElements:Int, data:UnsafeMutablePointer<Float32>)
{
    let ll =  numElements * MemoryLayout<Float32>.stride
    SnapShotsBuffer = device.makeBuffer(bytes:data, length: ll, options: MTLResourceOptions.storageModeManaged)
}

@_cdecl("EncodeSnapShots")
public func EncodeSnapShots(i:UInt32, j:UInt32){
    let SnapShotsCommandBuffer = commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = SnapShotsCommandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(Arguments_Buffer, offset: 0, index: 0)
    computeCommandEncoder.setComputePipelineState(computePipelineState_SnapShot)
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: 1), threadsPerThreadgroup:MTLSize(width:8, height:8, depth:1))
    computeCommandEncoder.endEncoding()
}

@_cdecl("EncodeSensors")
public func EncodeSensors(i:UInt32, j:UInt32, k:UInt32, x:UInt32, y:UInt32, z:UInt32){
    let SensorsCommandBuffer = commandQueue.makeCommandBuffer()!
    let computeCommandEncoder = SensorsCommandBuffer.makeComputeCommandEncoder()!
    computeCommandEncoder.setBuffer(Arguments_Buffer, offset: 0, index: 0)
    computeCommandEncoder.setComputePipelineState(computePipelineState_Sensors)
    computeCommandEncoder.dispatchThreadgroups(MTLSize(width: Int(i), height: Int(j), depth: Int(k)), threadsPerThreadgroup:MTLSize(width:Int(x), height:Int(y), depth:Int(z)))
    computeCommandEncoder.endEncoding()
    SensorsCommandBuffer.commit()
    SensorsCommandBuffer.waitUntilCompleted()

}

@_cdecl("SyncChange")
public func SyncChange(){
    let commandBufferSync = commandQueue.makeCommandBuffer()!
    let blitCommandEncoderSync: MTLBlitCommandEncoder = commandBufferSync.makeBlitCommandEncoder()!
    for buff in mex_buffer{
        blitCommandEncoderSync.synchronize(resource: buff!) 
    }
    blitCommandEncoderSync.endEncoding()
    commandBufferSync.commit()
    commandBufferSync.waitUntilCompleted()
    print("GPU and CPU Synced!")
}

@_cdecl("CopyFromGPUMEX")
public func CopyFromGPUMEX(index:UInt64) -> UnsafeMutablePointer<Float32>{
    return mex_buffer[mex_array[Int(index)]!]!.contents().assumingMemoryBound(to: Float32.self)

}
// @_cdecl("CopyFromGPUUInt")
// public func CopyFromGPUUInt() -> UnsafeMutablePointer<UInt32>{
//     return uint_buffer!.contents().assumingMemoryBound(to: UInt32.self)
// }

@_cdecl("CopyFromGpuSnapshot")
public func CopyFromGpuSnapshot() -> UnsafeMutablePointer<Float32>{
    return SnapShotsBuffer!.contents().assumingMemoryBound(to: Float32.self)
}

@_cdecl("maxThreadSensor")
public func maxThreadSensor() -> Int{
    return computePipelineState_Sensors.maxTotalThreadsPerThreadgroup
}

@_cdecl("freeGPUextern")
public func freeGPUextern() {
    // constant_buffer_uint!.setPurgeableState(MTLPurgeableState.empty)
    // constant_buffer_mex!.setPurgeableState(MTLPurgeableState.empty)
    for buff in mex_buffer{
        buff!.setPurgeableState(MTLPurgeableState.empty)
    }
    // uint_buffer!.setPurgeableState(MTLPurgeableState.empty)
    // index_mex!.setPurgeableState(MTLPurgeableState.empty)
    // index_uint!.setPurgeableState(MTLPurgeableState.empty)
    Arguments_Buffer!.setPurgeableState(MTLPurgeableState.empty)
    // SnapShotsBuffer!.setPurgeableState(MTLPurgeableState.empty)
}