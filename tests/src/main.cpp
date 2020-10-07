
#include <gtest/gtest.h>
#include <wx/fileconf.h>
#include <wx/stdpaths.h>

#include "Includes.h"

#ifndef UNIT_TESTING
#define UNIT_TESTING
#endif

int main(int argc, char **argv) {
    int resultTest = -2;

    try {
        ::testing::InitGoogleTest(&argc, argv);

        // Initialize the library because wxApp is not called
        wxInitialize();

        // Set the local config object
        wxFileConfig *pConfig = new wxFileConfig("FlHy", wxEmptyString,
                                                 wxStandardPaths::Get().GetTempDir() + "/FlHy.ini",
                                                 wxStandardPaths::Get().GetTempDir() + "/FlHy.ini",
                                                 wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);

        // Check path
        wxString filePath = wxFileName::GetCwd();
        wxString filePath1 = filePath;
        filePath1.Append("/tests/files");
        if (wxFileName::DirExists(filePath1)) {
            filePath.Append("/tests");
            wxSetWorkingDirectory(filePath);
        } else {
            wxString filePath2 = filePath;
            filePath2.Append("/../tests/files");
            if (wxFileName::DirExists(filePath2)) {
                filePath.Append("/../tests");
                wxSetWorkingDirectory(filePath);
            } else {
                wxString filePath3 = filePath;
                filePath3.Append("/../../tests/files");
                if (wxFileName::DirExists(filePath3)) {
                    filePath.Append("/../../tests");
                    wxSetWorkingDirectory(filePath);
                } else {
                    wxPrintf("Cannot find the files directory\n");
                    wxPrintf("Original working directory: %s\n", filePath);
                    return 0;
                }
            }
        }

        resultTest = RUN_ALL_TESTS();

        // Cleanup
        wxUninitialize();
        delete wxFileConfig::Set((wxFileConfig *)nullptr);

    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxPrintf(_("Exception caught: %s\n"), msg);
    }

    return resultTest;
}
