using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using UnityEngine;
internal static class NativeMethods {
    [DllImport ("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    internal static extern IntPtr LoadLibrary (
        string lpFileName
    );
    
}
static class DllLoder {
    private static IntPtr lib;

    public static void LoadNativeDll (string FileName) {
        if (lib != IntPtr.Zero) {
            return;
        }

        lib = NativeMethods.LoadLibrary (FileName);
        if (lib == IntPtr.Zero) {
            throw new Win32Exception ();
        }
    }
    
}